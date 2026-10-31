"""Phase 6, step 2: quantise the ONNX models to INT8.

    python scripts/24_quantize.py                     # all three
    python scripts/24_quantize.py --models detector --calib-method percentile

INT8 stores weights (and, for static quantisation, activations) as 8-bit integers with a
scale per channel: models get ~4x smaller and integer maths is what small ARM CPUs and
NPUs are fast at. The price is rounding error, measured in 25_edge_eval.py.

- detector, species (CNNs): static quantisation of the convolutions, signed 8-bit weights
  and activations (S8S8). Activation ranges are calibrated on images from TRAIN cameras
  only (never the cameras the models are tested on); box decoding and sigmoid stay float.
  Three things the detector needed (found by comparing layer outputs, see the README):
  * S8S8, not ONNX Runtime's old U8S8 default: on x86 CPUs without VNNI (this laptop's
    Ryzen 5600H) U8S8 can overflow a 16-bit intermediate and doubled the score error.
  * The detection heads stay float. Their class-score convolutions quantised worst by far
    (14-17 dB SQNR vs 30+ elsewhere); they are a few percent of the compute.
  * Percentile calibration (clip the top 0.001% of activations) instead of min/max, so a
    few outliers don't stretch the 256 levels. It keeps every activation in memory, so
    the detector calibrates on 48 frames (more ran out of RAM).
- species: DOESN'T WORK, kept for the record (25_edge_eval.py shows the damage). Features from
  the INT8 model have cosine ~0 with the FP32 ones and predictions agree ~20% of the time.
  Not one bad layer: each layer alone keeps 22-34 dB SQNR, normal for INT8, but EfficientNetV2
  has no layer that resets the error, so it compounds to ~7 dB by block 2 of 7. Keeping the
  SE and depthwise convolutions float (132 of 245) didn't help; dynamic INT8 was as wrong and
  15x slower (ConvInteger). Fixing it needs quantisation-aware training, and SpeciesNet's
  training data isn't available. The camera runs the species model in FP32 (configs/phase6.yaml).
- reid (a ViT): dynamic quantisation of the matrix multiplies. Weights are INT8 ahead of
  time, activation scales are computed per input at run time, so no calibration data is
  needed. This is the usual choice for transformers on CPU.

Writes outputs/phase6/onnx/<model>_int8.onnx (or <model>_int8_<method>.onnx for a
calibration method other than minmax).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ghats.edge import letterbox


def detector_calibration(n: int, seed: int) -> np.ndarray:
    """Letterboxed frames from train cameras, excluding the Phase 1 evaluation images."""
    m2 = pd.read_csv("data/swg/manifest_phase2.csv")
    m1 = set(pd.read_csv("data/swg/manifest.csv")["image_id"])
    train = set(pd.read_csv("data/crops/splits.csv").query("split == 'train'")["location"])
    pool = m2[m2["location"].isin(train) & ~m2["image_id"].isin(m1)]
    pick = pool.sample(n, random_state=seed)
    print(
        f"Detector calibration: {n} frames from {pick['location'].nunique()} train cameras, "
        f"{(pick['gt'] == 'empty').mean():.0%} empty"
    )
    out = []
    for p in pick["local_path"]:
        with Image.open(p) as im:
            out.append(letterbox(im)[0])
    return np.stack(out)


def species_calibration(n: int, seed: int) -> np.ndarray:
    """Tight detector crops from train cameras, with SpeciesNet's own preprocessing."""
    from ghats.baselines import TightCrops, speciesnet_preprocess

    crops = pd.read_csv("data/crops/crops.csv", dtype={"image_id": str})
    paths = pd.read_csv("data/swg/manifest_phase2.csv", dtype={"image_id": str}).set_index("image_id")["local_path"]
    pick = crops[(crops["split"] == "train") & crops["train_ok"]].sample(n, random_state=seed)
    print(f"Species calibration: {n} train crops, {pick['label'].nunique()} classes")
    ds = TightCrops(
        pick["image_id"].map(paths).tolist(), [ast.literal_eval(s) for s in pick["bbox"]], speciesnet_preprocess
    )
    return np.stack([ds[i].numpy() for i in range(len(ds))])


class Batches:
    """onnxruntime CalibrationDataReader over a stacked array."""

    def __init__(self, x: np.ndarray, batch: int = 8):
        self.it = iter([{"input": x[i : i + batch]} for i in range(0, len(x), batch)])

    def get_next(self):
        return next(self.it, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["detector", "reid", "species"])
    ap.add_argument("--dir", default="outputs/phase6/onnx")
    ap.add_argument("--n-calib", type=int, help="Default 48 for the detector, 256 for species")
    ap.add_argument(
        "--calib-method",
        choices=["minmax", "percentile", "entropy"],
        help="Default percentile for the detector, minmax for species",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    d = Path(args.dir)
    methods = {
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
    }
    defaults = {"detector": ("percentile", 48), "species": ("minmax", 256), "reid": (None, 0)}
    for name in args.models:
        method = args.calib_method or defaults[name][0]
        n_calib = args.n_calib or defaults[name][1]
        suffix = "" if args.calib_method is None else f"_{method}"
        src, pre, dst = d / f"{name}_fp32.onnx", d / f"{name}_pre.onnx", d / f"{name}_int8{suffix}.onnx"
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
        if name == "reid":
            quantize_dynamic(
                str(pre), str(dst), op_types_to_quantize=["MatMul"], per_channel=True, weight_type=QuantType.QInt8
            )
        else:
            x = (detector_calibration if name == "detector" else species_calibration)(n_calib, args.seed)
            graph = onnx.load(str(pre), load_external_data=False).graph
            keep_float = [n.name for n in graph.node if n.op_type == "Conv" and "/heads." in n.name]
            quantize_static(
                str(pre),
                str(dst),
                Batches(x, 4),
                quant_format=QuantFormat.QDQ,
                op_types_to_quantize=["Conv"],
                nodes_to_exclude=keep_float if name == "detector" else [],
                per_channel=True,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=methods[method],
                extra_options={"CalibMovingAverage": False},
            )
        pre.unlink()
        print(f"{name}: {src.stat().st_size / 2**20:.1f} MB -> {dst.stat().st_size / 2**20:.1f} MB ({dst.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
