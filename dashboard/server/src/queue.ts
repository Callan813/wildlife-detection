// Mixing the random audit sample into the uncertain-first queue.
//
// Uncertain-first review spends reviewers' time where the model is least sure, but those
// items all sit near the decision boundary, so they say little about how good the model
// is overall or where the "new leopard" threshold belongs. Every `every`-th slot goes to
// an audit item (a random sample fixed at export), and only audit reviews are used to
// measure the models. Reviewers aren't told which is which.

export const AUDIT_EVERY = 5;

export function interleave<T>(uncertain: T[], audit: T[], limit: number, every = AUDIT_EVERY): T[] {
  const out: T[] = [];
  let u = 0;
  let a = 0;
  while (out.length < limit && (u < uncertain.length || a < audit.length)) {
    const auditSlot = (out.length + 1) % every === 0;
    if ((auditSlot && a < audit.length) || u >= uncertain.length) out.push(audit[a++]);
    else out.push(uncertain[u++]);
  }
  return out;
}
