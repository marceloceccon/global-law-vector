/**
 * Seeded pseudo-random numbers, uniform in [0, 1).
 *
 * A 32-bit linear congruential generator (Numerical Recipes constants). Not for anything
 * statistical beyond shuffling and jitter, but deterministic: the same seed gives the same
 * sample order, PCA start vectors and point sizes on every run. Use the full value (scale it,
 * then floor it), never `state % k`: the low bits of a power-of-two LCG have short periods.
 */
export function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Fisher–Yates shuffle in place, driven by `rand`. */
export function shuffle<T>(items: T[], rand: () => number): T[] {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}
