"""Phase 6, step 1: export the three models to ONNX and check they match PyTorch.

    python scripts/23_export_onnx.py                  # all three
    python scripts/23_export_onnx.py --models detector

ONNX is the hand-off format for edge runtimes: ONNX Runtime on a Raspberry Pi, TensorRT on
a Jetson. Each model is exported in FP32 and run on the same random batch in both PyTorch
and ONNX Runtime; the largest output difference is printed and saved, so a broken export
shows up here and not as a mysterious accuracy drop later.

- detector: MegaDetector V6 (YOLOv9-c, MIT). The graph ends at per-anchor class scores
  (sigmoid applied) and decoded boxes in 640x640 letterbox pixels. NMS and the letterbox
  stay outside, in numpy (ghats.edge), as they would on the device.
- reid: the ArcFace-trained DINOv2-S (outputs/phase3/reid_lp102_s1), 224x448 crop in,
  512-d embedding out (not normalised; the caller normalises).
- species: SpeciesNet's backbone cut at the pooled 1,280-d features that the Phase 2
  probe uses. The probe itself is a logistic regression, applied in numpy.

Writes outputs/phase6/onnx/{detector,reid,species}_fp32.onnx and export_check.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ghats.config import load_config

OPSET = 17


class DetectorHead(nn.Module):
    """YOLOv9 'Main' head outputs -> (scores [B, N, 3], boxes [B, N, 4]), as PyTorch-Wildlife's Vec2Box."""

    def __init__(self, model: nn.Module, anchor_grid: torch.Tensor, scaler: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("grid", anchor_grid.clone())
        self.register_buffer("scaler", scaler.clone())

    def forward(self, x):
        cls, box = [], []
        for c, _, b in self.model(x)["Main"]:
            n, k, h, w = c.shape
            cls.append(c.permute(0, 2, 3, 1).reshape(n, h * w, k))
            box.append(b.permute(0, 2, 3, 1).reshape(n, h * w, 4))
        lt, rb = (torch.cat(box, 1) * self.scaler.view(1, -1, 1)).chunk(2, -1)
        return torch.cat(cls, 1).sigmoid(), torch.cat([self.grid - lt, self.grid + rb], -1)


def detector(cfg: dict) -> tuple[nn.Module, torch.Tensor, list[str]]:
    from ghats.detector import MegaDetector

    d = cfg["detector"]
    md = MegaDetector(d["class"], d["version"], device="cpu")
    conv = md.model.post_proccess.converter
    return (
        DetectorHead(md.model.model, conv.anchor_grid, conv.scaler).eval(),
        torch.rand(2, 3, 640, 640),
        [
            "scores",
            "boxes",
        ],
    )


def reid(cfg: dict, weights: Path) -> tuple[nn.Module, torch.Tensor, list[str]]:
    from ghats.reid_model import load_trained

    h, w = cfg["train"]["size"]
    model = load_trained(weights, "cpu")
    # Built with dynamic_img_size, the backbone resizes its position embeddings on every call
    # (antialiased bicubic, which ONNX can't express). Crops are always h x w, so resize once
    # here; the outputs are bit-identical.
    model.backbone.set_input_size(img_size=(h, w))
    return model, torch.randn(2, 3, h, w), ["embedding"]


def species(cfg: dict) -> tuple[nn.Module, torch.Tensor, list[str]]:
    """SpeciesNet (a torch.fx graph from onnx2torch) with its output moved to the pooled features."""
    from ghats.baselines import SPECIESNET_POOL_NODE, SPECIESNET_SIZE

    b = cfg["baselines"]["speciesnet"]
    # weights_only=False: the checkpoint is a pickled fx graph (approved for SpeciesNet in Phase 2).
    gm = torch.load(Path(b["dir"]) / b["weights"], map_location="cpu", weights_only=False).eval()
    pool = next(n for n in gm.graph.nodes if n.op == "call_module" and SPECIESNET_POOL_NODE in n.name)
    out = next(n for n in gm.graph.nodes if n.op == "output")
    out.args = (pool,)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm, torch.rand(2, SPECIESNET_SIZE, SPECIESNET_SIZE, 3), ["features"]


def export(model: nn.Module, x: torch.Tensor, outputs: list[str], path: Path) -> dict:
    import onnxruntime as ort

    torch.onnx.export(
        model,
        (x,),
        str(path),
        input_names=["input"],
        output_names=outputs,
        dynamic_axes={k: {0: "batch"} for k in ["input", *outputs]},
        opset_version=OPSET,
        dynamo=False,
    )
    with torch.no_grad():
        ref = model(x)
    ref = [ref] if isinstance(ref, torch.Tensor) else list(ref)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})
    diffs = {name: float(np.abs(g - r.numpy()).max()) for name, g, r in zip(outputs, got, ref)}
    scale = {name: float(np.abs(r.numpy()).max()) for name, r in zip(outputs, ref)}
    return {"file": path.name, "mb": round(path.stat().st_size / 2**20, 1), "max_abs_diff": diffs, "max_abs": scale}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["detector", "reid", "species"])
    ap.add_argument("--reid-weights", default="outputs/phase3/reid_lp102_s1/best.pt")
    ap.add_argument("--out", default="outputs/phase6/onnx")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    check_path = out / "export_check.json"
    checks = json.loads(check_path.read_text()) if check_path.exists() else {}
    build = {
        "detector": lambda: detector(load_config("configs/phase1.yaml")),
        "reid": lambda: reid(load_config("configs/phase3.yaml"), Path(args.reid_weights)),
        "species": lambda: species(load_config("configs/phase2.yaml")),
    }
    for name in args.models:
        model, x, outputs = build[name]()
        params = sum(p.numel() for p in model.parameters())
        checks[name] = export(model, x, outputs, out / f"{name}_fp32.onnx") | {"params_m": round(params / 1e6, 2)}
        print(name, json.dumps(checks[name]))
    check_path.write_text(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
