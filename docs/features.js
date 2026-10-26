// Exact port of common.py. Keep in sync!
export const FPS = 15, WIN = 30;
const IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
export const N_FEAT = IDX.length * 2 + 5;

export function frameFeatures(lm, w, h) {
  const px = lm.map(p => [p.x * w, p.y * h]);
  const hip = [(px[23][0] + px[24][0]) / 2, (px[23][1] + px[24][1]) / 2];
  const sh = [(px[11][0] + px[12][0]) / 2, (px[11][1] + px[12][1]) / 2];
  const v = [sh[0] - hip[0], sh[1] - hip[1]];
  const torso = Math.max(Math.hypot(v[0], v[1]), 1e-3);
  const out = [];
  for (const i of IDX) out.push((px[i][0] - hip[0]) / torso, (px[i][1] - hip[1]) / torso);
  const xs = IDX.map(i => px[i][0]), ys = IDX.map(i => px[i][1]);
  const bw = Math.max(...xs) - Math.min(...xs);
  const bh = Math.max(Math.max(...ys) - Math.min(...ys), 1e-3);
  const aspect = Math.min(Math.max(bw / bh, 0), 5);
  out.push(hip[0] / w, hip[1] / h, v[0] / torso, -v[1] / torso, aspect);
  return Float32Array.from(out);
}

// win: array of WIN Float32Array(N_FEAT) -> Float32Array [2*N_FEAT, WIN] channel-major
export function toInput(win) {
  const data = new Float32Array(2 * N_FEAT * WIN);
  for (let t = 0; t < WIN; t++) for (let c = 0; c < N_FEAT; c++) {
    data[c * WIN + t] = win[t][c];
    data[(N_FEAT + c) * WIN + t] = t === 0 ? 0 : win[t][c] - win[t - 1][c];
  }
  return data;
}