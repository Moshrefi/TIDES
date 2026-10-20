// Python bindings + CPU backend of the fused time-tangent rasterizer.
#include <torch/extension.h>

#include "tangent_raster.h"

#ifdef TIDES_WITH_CUDA
void tangent_raster_cuda(const torch::Tensor& feats, const torch::Tensor& gid, const torch::Tensor& ranges,
                         const torch::Tensor& tiles, torch::Tensor& out, int width, int height, int tile,
                         float tau_max, float alpha_min, float T_min);
#endif

static void tangent_raster_cpu(const torch::Tensor& feats, const torch::Tensor& gid, const torch::Tensor& ranges,
                               const torch::Tensor& tiles, torch::Tensor& out, int width, int height, int tile,
                               float tau_max, float alpha_min, float T_min) {
  const float* F = feats.data_ptr<float>();
  const int* G = gid.data_ptr<int>();
  const int* R = ranges.data_ptr<int>();
  const int* TL = tiles.data_ptr<int>();
  float* O = out.data_ptr<float>();
  const int n_tx = (width + tile - 1) / tile;
  const int n = tiles.size(0);
  at::parallel_for(0, n, 1, [&](int64_t b0, int64_t b1) {
    for (int64_t b = b0; b < b1; ++b) {
      const int t = TL[b];
      const int tx = t % n_tx, ty = t / n_tx;
      const int start = R[2 * t], end = R[2 * t + 1];
      for (int oy = 0; oy < tile; ++oy) {
        for (int ox = 0; ox < tile; ++ox) {
          const int x = tx * tile + ox, y = ty * tile + oy;
          if (x >= width || y >= height) continue;
          PixelState s;
          pixel_init(s);
          const float px = x + 0.5f, py = y + 0.5f;
          for (int k = start; k < end; ++k) {
            if (!blend_step(s, F + (int64_t)G[k] * F_NUM, px, py, tau_max, alpha_min, T_min)) break;
          }
          pixel_write(s, O + ((int64_t)y * width + x) * O_NUM);
        }
      }
    }
  });
}

// feats: (N, F_NUM) float32; gid: (M,) int32 Gaussian ids sorted by (tile, depth);
// ranges: (n_tiles, 2) int32 [start, end) into gid; tiles: (n,) int32 tiles to render.
// Returns out: (H*W, O_NUM); pixels of tiles not rendered keep T=1 and zeros.
torch::Tensor tangent_raster(torch::Tensor feats, torch::Tensor gid, torch::Tensor ranges, torch::Tensor tiles,
                             int64_t width, int64_t height, int64_t tile, double tau_max, double alpha_min,
                             double T_min) {
  TORCH_CHECK(feats.dtype() == torch::kFloat32 && feats.size(1) == F_NUM, "feats must be float32 (N, 18)");
  feats = feats.contiguous();
  gid = gid.to(torch::kInt32).contiguous();
  ranges = ranges.to(torch::kInt32).contiguous();
  tiles = tiles.to(torch::kInt32).contiguous();
  auto out = torch::zeros({height * width, (int64_t)O_NUM}, feats.options());
  out.select(1, O_T).fill_(1.f);
  if (tiles.numel() == 0) return out;
  if (feats.is_cuda()) {
#ifdef TIDES_WITH_CUDA
    TORCH_CHECK(tile * tile <= 1024, "tile*tile must be <= 1024 on CUDA");
    tangent_raster_cuda(feats, gid, ranges, tiles, out, width, height, tile, tau_max, alpha_min, T_min);
#else
    TORCH_CHECK(false, "built without CUDA");
#endif
  } else {
    tangent_raster_cpu(feats, gid, ranges, tiles, out, width, height, tile, tau_max, alpha_min, T_min);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tangent_raster", &tangent_raster, "fused time-tangent Gaussian rasterizer");
}
