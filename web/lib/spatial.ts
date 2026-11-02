/**
 * Local density of the projected cloud, computed in the browser from the coordinates alone.
 * The renderer uses it to strengthen the faint halo where points crowd together.
 *
 * Density is the number of neighbours within a fixed radius. The radius is the mean spacing
 * a uniform cloud of the same size would have in the same bounding box, so the counts stay in
 * a similar range whatever the sample size. A uniform grid with cell size = radius, filled by
 * counting sort, keeps the pass linear: 40,000 points take well under a second.
 */

/** Cells per axis at most; bounds memory when the cloud is very flat along one axis. */
const MAX_CELLS_PER_AXIS = 256;

/** Neighbour count per point within the search radius, scaled so the 95th percentile is 1. */
export function localDensity(xyz: ArrayLike<number>, n: number): Float32Array {
  const density = new Float32Array(n);
  if (n === 0) return density;

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = xyz[i * 3], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const spanX = maxX - minX, spanY = maxY - minY, spanZ = maxZ - minZ;
  const span = Math.max(spanX, spanY, spanZ, 1e-6);
  const volume = Math.max(spanX * spanY * spanZ, (span / MAX_CELLS_PER_AXIS) ** 3);
  const radius = Math.max(Math.cbrt(volume / n) * 1.4, span / MAX_CELLS_PER_AXIS);
  const r2 = radius * radius;

  const gx = Math.min(MAX_CELLS_PER_AXIS, Math.floor(spanX / radius) + 1);
  const gy = Math.min(MAX_CELLS_PER_AXIS, Math.floor(spanY / radius) + 1);
  const gz = Math.min(MAX_CELLS_PER_AXIS, Math.floor(spanZ / radius) + 1);
  const cell = (v: number, min: number, g: number) => Math.min(g - 1, Math.floor((v - min) / radius));

  // Counting sort of point indices by cell: items[start[c] .. start[c + 1]) are in cell c.
  const cellOf = new Int32Array(n);
  const start = new Int32Array(gx * gy * gz + 1);
  for (let i = 0; i < n; i++) {
    const c = cell(xyz[i * 3], minX, gx) + gx * (cell(xyz[i * 3 + 1], minY, gy) + gy * cell(xyz[i * 3 + 2], minZ, gz));
    cellOf[i] = c;
    start[c + 1]++;
  }
  for (let c = 0; c < start.length - 1; c++) start[c + 1] += start[c];
  const fill = new Int32Array(start.length);
  const items = new Int32Array(n);
  for (let i = 0; i < n; i++) items[start[cellOf[i]] + fill[cellOf[i]]++] = i;

  for (let i = 0; i < n; i++) {
    const x = xyz[i * 3], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
    const cx = cell(x, minX, gx), cy = cell(y, minY, gy), cz = cell(z, minZ, gz);
    let count = 0;
    for (let zz = Math.max(0, cz - 1); zz <= Math.min(gz - 1, cz + 1); zz++) {
      for (let yy = Math.max(0, cy - 1); yy <= Math.min(gy - 1, cy + 1); yy++) {
        for (let xx = Math.max(0, cx - 1); xx <= Math.min(gx - 1, cx + 1); xx++) {
          const c = xx + gx * (yy + gy * zz);
          for (let p = start[c]; p < start[c + 1]; p++) {
            const j = items[p];
            const dx = xyz[j * 3] - x, dy = xyz[j * 3 + 1] - y, dz = xyz[j * 3 + 2] - z;
            if (j !== i && dx * dx + dy * dy + dz * dz <= r2) count++;
          }
        }
      }
    }
    density[i] = count;
  }

  // Scale by the 95th percentile so a few extreme cores do not flatten everything else.
  const sorted = Float32Array.from(density).sort();
  const p95 = Math.max(sorted[Math.floor(0.95 * (n - 1))], 1);
  for (let i = 0; i < n; i++) density[i] = Math.min(density[i] / p95, 1);
  return density;
}
