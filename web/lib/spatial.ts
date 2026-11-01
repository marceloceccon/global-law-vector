/**
 * Client-side neighbourhood analysis of the projected cloud, used purely for looks:
 * local density drives brightness, size and haze; nearest neighbours become filaments.
 *
 * A uniform grid (cell = search radius) built with a counting sort keeps this linear in the
 * number of points: 40,000 points take about 100 ms in the browser. Everything is in the
 * 3-D projected space, so it needs nothing from the server beyond the coordinates.
 */

export type Neighbourhood = {
  /** Local density per point, scaled so the 95th percentile is 1. */
  density: Float32Array;
  /** Undirected edges as index pairs [i0, j0, i1, j1, ...], each point to its k nearest. */
  edges: Uint32Array;
  radius: number;
};

export function analyse(xyz: ArrayLike<number>, n: number, k = 2): Neighbourhood {
  if (n === 0) return { density: new Float32Array(0), edges: new Uint32Array(0), radius: 0 };

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = xyz[i * 3], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  // Radius from the mean spacing a uniform cloud of this size would have in this box, so
  // the number of neighbours stays in the same range whatever the sample size.
  const volume = Math.max((maxX - minX) * (maxY - minY) * (maxZ - minZ), 1e-9);
  const radius = Math.cbrt(volume / n) * 1.4;
  const r2 = radius * radius;

  const gx = Math.max(1, Math.ceil((maxX - minX) / radius) + 1);
  const gy = Math.max(1, Math.ceil((maxY - minY) / radius) + 1);
  const gz = Math.max(1, Math.ceil((maxZ - minZ) / radius) + 1);
  const cellOf = new Int32Array(n);
  const counts = new Int32Array(gx * gy * gz + 1);
  for (let i = 0; i < n; i++) {
    const cx = Math.floor((xyz[i * 3] - minX) / radius);
    const cy = Math.floor((xyz[i * 3 + 1] - minY) / radius);
    const cz = Math.floor((xyz[i * 3 + 2] - minZ) / radius);
    const c = cx + gx * (cy + gy * cz);
    cellOf[i] = c;
    counts[c + 1]++;
  }
  for (let c = 0; c < counts.length - 1; c++) counts[c + 1] += counts[c];
  const start = counts; // start[c] .. start[c+1] index into `items`
  const fill = new Int32Array(start.length);
  const items = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const c = cellOf[i];
    items[start[c] + fill[c]++] = i;
  }

  const density = new Float32Array(n);
  const edgeSet = new Set<number>();
  const bestD = new Float64Array(k);
  const bestJ = new Int32Array(k);
  for (let i = 0; i < n; i++) {
    const x = xyz[i * 3], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
    const cx = Math.floor((x - minX) / radius);
    const cy = Math.floor((y - minY) / radius);
    const cz = Math.floor((z - minZ) / radius);
    let found = 0;
    bestD.fill(Infinity);
    let count = 0;
    for (let dz = -1; dz <= 1; dz++) {
      const zz = cz + dz;
      if (zz < 0 || zz >= gz) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = cy + dy;
        if (yy < 0 || yy >= gy) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = cx + dx;
          if (xx < 0 || xx >= gx) continue;
          const c = xx + gx * (yy + gy * zz);
          for (let p = start[c]; p < start[c + 1]; p++) {
            const j = items[p];
            if (j === i) continue;
            const ex = xyz[j * 3] - x, ey = xyz[j * 3 + 1] - y, ez = xyz[j * 3 + 2] - z;
            const d2 = ex * ex + ey * ey + ez * ez;
            if (d2 > r2) continue;
            count++;
            if (d2 < bestD[k - 1]) {
              // insert into the small sorted list of nearest
              let q = Math.min(found, k - 1);
              while (q > 0 && bestD[q - 1] > d2) {
                bestD[q] = bestD[q - 1];
                bestJ[q] = bestJ[q - 1];
                q--;
              }
              bestD[q] = d2;
              bestJ[q] = j;
              if (found < k) found++;
            }
          }
        }
      }
    }
    density[i] = count;
    for (let q = 0; q < found; q++) {
      const j = bestJ[q];
      edgeSet.add(i < j ? i * n + j : j * n + i);
    }
  }

  // Normalise density by its 95th percentile so a few extreme cores do not flatten the rest.
  const sorted = Float32Array.from(density).sort();
  const p95 = Math.max(sorted[Math.floor(0.95 * (n - 1))], 1);
  for (let i = 0; i < n; i++) density[i] = Math.min(density[i] / p95, 1);

  const edges = new Uint32Array(edgeSet.size * 2);
  let e = 0;
  for (const key of edgeSet) {
    edges[e++] = Math.floor(key / n);
    edges[e++] = key % n;
  }
  return { density, edges, radius };
}
