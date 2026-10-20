// Shared per-pixel compositing step for the time-tangent rasterizer.
// Compiled both for the CPU (C++) and the GPU (CUDA) backends, so the two
// backends execute exactly the same arithmetic as the PyTorch reference
// (tides/render/rasterizer.py), Eq. 9-14 of the paper.
#pragma once

#ifdef __CUDACC__
#define TIDES_HD __host__ __device__ __forceinline__
#else
#define TIDES_HD inline
#endif

#include <cmath>

// Packed per-Gaussian features (row of the `feats` tensor).
enum Feat {
  F_U = 0, F_V, F_UD, F_VD,
  F_QA, F_QB, F_QC, F_QAD, F_QBD, F_QCD,
  F_OP, F_OPD,
  F_R, F_G, F_B, F_RD, F_GD, F_BD,
  F_NUM
};

// Per-pixel output channels.
enum Out {
  O_R = 0, O_G, O_B, O_RD, O_GD, O_BD, O_T, O_TD, O_CNT, O_BSUM, O_TSUM,
  O_NUM
};

struct PixelState {
  float T, Td;
  float acc[3], accd[3];
  float cnt, bsum, tsum;
  bool done;
};

TIDES_HD void pixel_init(PixelState& s) {
  s.T = 1.f; s.Td = 0.f;
  for (int c = 0; c < 3; ++c) { s.acc[c] = 0.f; s.accd[c] = 0.f; }
  s.cnt = 0.f; s.bsum = 0.f; s.tsum = 0.f; s.done = false;
}

// Composite one Gaussian (front-to-back) into the pixel state.
// Returns false when the pixel has terminated (transmittance would drop below T_min).
TIDES_HD bool blend_step(PixelState& s, const float* f, float px, float py,
                         float tau_max, float alpha_min, float T_min) {
  const float dx = px - f[F_U];
  const float dy = py - f[F_V];
  const float qa = f[F_QA], qb = f[F_QB], qc = f[F_QC];
  // Eq. 9
  const float power = -0.5f * (qa * dx * dx + 2.f * qb * dx * dy + qc * dy * dy);
  if (power > 0.f) return true;
  const float g = expf(power);
  const float op = f[F_OP];
  const float tau_raw = op * g;  // Eq. 2
  const bool sat = tau_raw >= tau_max;
  const float tau = sat ? tau_max : tau_raw;
  if (tau < alpha_min) return true;
  const float test_T = s.T * (1.f - tau);
  if (test_T < T_min) { s.done = true; return false; }

  // Eq. 10: d_dot = -u_dot
  const float ddx = -f[F_UD], ddy = -f[F_VD];
  const float pd = -0.5f * ((f[F_QAD] * dx * dx + 2.f * f[F_QBD] * dx * dy + f[F_QCD] * dy * dy) +
                            2.f * (ddx * (qa * dx + qb * dy) + ddy * (qb * dx + qc * dy)));
  const float gd = g * pd;
  // Eq. 11: no derivative through the saturation clamp
  const float taud = sat ? 0.f : (f[F_OPD] * g + op * gd);

  // Eq. 12
  const float w = s.T * tau;
  const float wd = s.Td * tau + s.T * taud;
  s.acc[0] += w * f[F_R];  s.acc[1] += w * f[F_G];  s.acc[2] += w * f[F_B];
  s.accd[0] += wd * f[F_R] + w * f[F_RD];
  s.accd[1] += wd * f[F_G] + w * f[F_GD];
  s.accd[2] += wd * f[F_B] + w * f[F_BD];
  s.cnt += 1.f; s.bsum += w; s.tsum += tau;
  // Eq. 13
  s.Td = s.Td * (1.f - tau) - s.T * taud;
  s.T = test_T;
  return true;
}

TIDES_HD void pixel_write(const PixelState& s, float* o) {
  o[O_R] = s.acc[0]; o[O_G] = s.acc[1]; o[O_B] = s.acc[2];
  o[O_RD] = s.accd[0]; o[O_GD] = s.accd[1]; o[O_BD] = s.accd[2];
  o[O_T] = s.T; o[O_TD] = s.Td; o[O_CNT] = s.cnt; o[O_BSUM] = s.bsum; o[O_TSUM] = s.tsum;
}
