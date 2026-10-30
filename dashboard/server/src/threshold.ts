// Recalibrating the "new leopard" threshold from reviewer decisions.
//
// The starting threshold was picked on validation leopards, and Phase 3 showed it doesn't
// transfer well to a different set of leopards. Every review is a label: "match" means
// the leopard was in the catalogue, "new" means it wasn't. With enough of both, pick the
// threshold on the model's best-match similarity that agrees with reviewers best
// (balanced accuracy, so the more common decision doesn't dominate).

export interface Labelled {
  bestSim: number;
  isNew: boolean;
}

export interface Calibration {
  threshold: number;
  balancedAccuracy: number;
  matches: number;
  news: number;
}

export const MIN_EACH = 10;

export function calibrate(items: Labelled[], grid = defaultGrid()): Calibration | null {
  const news = items.filter((i) => i.isNew);
  const matches = items.filter((i) => !i.isNew);
  if (news.length < MIN_EACH || matches.length < MIN_EACH) return null;

  let best: Calibration | null = null;
  for (const t of grid) {
    // Below the threshold the model says "new".
    const newRight = news.filter((i) => i.bestSim < t).length / news.length;
    const matchRight = matches.filter((i) => i.bestSim >= t).length / matches.length;
    const balancedAccuracy = (newRight + matchRight) / 2;
    if (!best || balancedAccuracy > best.balancedAccuracy) {
      best = { threshold: t, balancedAccuracy, matches: matches.length, news: news.length };
    }
  }
  return best;
}

function defaultGrid(): number[] {
  return Array.from({ length: 101 }, (_, i) => Math.round((i * 0.01) * 100) / 100);
}
