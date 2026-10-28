// student_kernel.hpp
// Optimized Canny edge detection with OpenMP
//
// - Symmetric 5-tap Gaussian passes with a sliding window
// - Fused Sobel, magnitude, and direction quantization
// - Non-maximum suppression with a direction lookup table
// - Hysteresis as a parallel wavefront BFS with compare-and-swap
//
// Compile: g++ -fopenmp -march=native main.cpp -o canny

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <omp.h>
#include <vector>

namespace student {

// ============================================================================
// GAUSSIAN 5x1 (Horizontal): Symmetric Kernel Optimization
// ============================================================================
static void gaussian5x1(const std::vector<uint8_t> &__restrict in,
                        std::vector<float> &__restrict tmp, int W, int H) {
  constexpr float k_outer = 1.0f;
  constexpr float k_mid = 4.0f;
  constexpr float k_center = 6.0f;
  constexpr float norm = 1.0f / 16.0f;

#pragma omp parallel for schedule(static)
  for (int y = 0; y < H; ++y) {
    const uint8_t *src_row = &in[(size_t)y * W];
    float *dst_row = &tmp[(size_t)y * W];

    if (W >= 4) {
      float i0 = src_row[0], i1 = src_row[1], i2 = src_row[2], i3 = src_row[3];
      float s0 = k_outer * (i0 + i2) + k_mid * (i0 + i1) + k_center * i0;
      float s1 = k_outer * (i0 + i3) + k_mid * (i0 + i2) + k_center * i1;
      dst_row[0] = s0 * norm;
      dst_row[1] = s1 * norm;
    }

    if (W >= 5) {
      float i0 = src_row[0], i1 = src_row[1], i2 = src_row[2];
      float i3 = src_row[3], i4 = src_row[4];
      for (int x = 2; x < W - 2; ++x) {
        float sum = k_outer * (i0 + i4) + k_mid * (i1 + i3) + k_center * i2;
        dst_row[x] = sum * norm;
        i0 = i1;
        i1 = i2;
        i2 = i3;
        i3 = i4;
        i4 = src_row[x + 3];
      }
    }

    if (W >= 4) {
      float i1_r = src_row[W - 4], i2_r = src_row[W - 3];
      float i3_r = src_row[W - 2], i4_r = src_row[W - 1];
      float s0 =
          k_outer * (i1_r + i4_r) + k_mid * (i2_r + i4_r) + k_center * i3_r;
      float s1 =
          k_outer * (i2_r + i4_r) + k_mid * (i3_r + i4_r) + k_center * i4_r;
      dst_row[W - 2] = s0 * norm;
      dst_row[W - 1] = s1 * norm;
    }
  }
}

// ============================================================================
// GAUSSIAN 1x5 (Vertical)
// ============================================================================
static void gaussian1x5_from_tmp(const std::vector<float> &__restrict tmp,
                                 std::vector<float> &__restrict out, int W,
                                 int H) {
  constexpr float k0 = 1.0f, k1 = 4.0f, k2 = 6.0f;
  constexpr float norm = 1.0f / 16.0f;

  const float *src = tmp.data();
  float *dst = out.data();

  // Top boundary rows - parallelize for 8k+ images
  for (int y = 0; y < std::min(2, H); ++y) {
    const float *row0 = src + (size_t)std::max(0, y - 2) * W;
    const float *row1 = src + (size_t)std::max(0, y - 1) * W;
    const float *row2 = src + (size_t)y * W;
    const float *row3 = src + (size_t)std::min(H - 1, y + 1) * W;
    const float *row4 = src + (size_t)std::min(H - 1, y + 2) * W;
    float *dst_row = dst + (size_t)y * W;
#pragma omp parallel for schedule(static) if (W >= 8000)
    for (int x = 0; x < W; ++x) {
      dst_row[x] = (k0 * row0[x] + k1 * row1[x] + k2 * row2[x] + k1 * row3[x] +
                    k0 * row4[x]) *
                   norm;
    }
  }

  if (H >= 5) {
#pragma omp parallel for schedule(static)
    for (int y = 2; y < H - 2; ++y) {
      const float *row0 = src + (size_t)(y - 2) * W;
      const float *row1 = src + (size_t)(y - 1) * W;
      const float *row2 = src + (size_t)y * W;
      const float *row3 = src + (size_t)(y + 1) * W;
      const float *row4 = src + (size_t)(y + 2) * W;
      float *dst_row = dst + (size_t)y * W;
      for (int x = 0; x < W; ++x) {
        dst_row[x] = (k0 * row0[x] + k1 * row1[x] + k2 * row2[x] +
                      k1 * row3[x] + k0 * row4[x]) *
                     norm;
      }
    }
  }

  // Bottom boundary rows - parallelize for 8k+ images
  for (int y = std::max(2, H - 2); y < H; ++y) {
    const float *row0 = src + (size_t)std::max(0, y - 2) * W;
    const float *row1 = src + (size_t)std::max(0, y - 1) * W;
    const float *row2 = src + (size_t)y * W;
    const float *row3 = src + (size_t)std::min(H - 1, y + 1) * W;
    const float *row4 = src + (size_t)std::min(H - 1, y + 2) * W;
    float *dst_row = dst + (size_t)y * W;
#pragma omp parallel for schedule(static) if (W >= 8000)
    for (int x = 0; x < W; ++x) {
      dst_row[x] = (k0 * row0[x] + k1 * row1[x] + k2 * row2[x] + k1 * row3[x] +
                    k0 * row4[x]) *
                   norm;
    }
  }
}

// ============================================================================
// FUSED SOBEL + GRADIENT
// ============================================================================
static void sobel_and_grad(const std::vector<float> &in,
                           std::vector<float> &mag, std::vector<uint8_t> &dir,
                           int W, int H) {
  constexpr float TAN_22_5 = 0.41421356f;
  constexpr float TAN_67_5 = 2.41421356f;

  const float *src = in.data();
  float *p_mag = mag.data();
  uint8_t *p_dir = dir.data();

  if (H < 2)
    return;

  auto process_row = [&](const float *p_up, const float *p_mid,
                         const float *p_dn, int y) {
    float u0 = p_up[0], m0 = p_mid[0], d0 = p_dn[0];
    float u1 = p_up[0], m1 = p_mid[0], d1 = p_dn[0];
    float u2 = p_up[1], m2 = p_mid[1], d2 = p_dn[1];

    {
      float gx = (u2 + 2.0f * m2 + d2) - (u0 + 2.0f * m0 + d0);
      float gy = (d0 + 2.0f * d1 + d2) - (u0 + 2.0f * u1 + u2);
      p_mag[y * W] = __builtin_sqrtf(gx * gx + gy * gy);
      float abs_x = __builtin_fabsf(gx), abs_y = __builtin_fabsf(gy);
      uint8_t q = (abs_y <= abs_x * TAN_22_5)  ? 0
                  : (abs_y > abs_x * TAN_67_5) ? 90
                  : ((gx > 0) == (gy > 0))     ? 45
                                               : 135;
      p_dir[y * W] = q;
    }

    for (int x = 1; x < W - 1; ++x) {
      u0 = u1;
      m0 = m1;
      d0 = d1;
      u1 = u2;
      m1 = m2;
      d1 = d2;
      u2 = p_up[x + 1];
      m2 = p_mid[x + 1];
      d2 = p_dn[x + 1];

      float gx = (u2 + 2.0f * m2 + d2) - (u0 + 2.0f * m0 + d0);
      float gy = (d0 + 2.0f * d1 + d2) - (u0 + 2.0f * u1 + u2);
      int idx = y * W + x;
      p_mag[idx] = __builtin_sqrtf(gx * gx + gy * gy);
      float abs_x = __builtin_fabsf(gx), abs_y = __builtin_fabsf(gy);
      uint8_t q = (abs_y <= abs_x * TAN_22_5)  ? 0
                  : (abs_y > abs_x * TAN_67_5) ? 90
                  : ((gx > 0) == (gy > 0))     ? 45
                                               : 135;
      p_dir[idx] = q;
    }

    {
      u0 = u1;
      m0 = m1;
      d0 = d1;
      u1 = u2;
      m1 = m2;
      d1 = d2;
      u2 = u1;
      m2 = m1;
      d2 = d1;
      float gx = (u2 + 2 * m2 + d2) - (u0 + 2 * m0 + d0);
      float gy = (d0 + 2 * d1 + d2) - (u0 + 2 * u1 + u2);
      int idx = y * W + W - 1;
      p_mag[idx] = __builtin_sqrtf(gx * gx + gy * gy);
      float abs_x = __builtin_fabsf(gx), abs_y = __builtin_fabsf(gy);
      uint8_t q = (abs_y <= abs_x * TAN_22_5)  ? 0
                  : (abs_y > abs_x * TAN_67_5) ? 90
                  : ((gx > 0) == (gy > 0))     ? 45
                                               : 135;
      p_dir[idx] = q;
    }
  };

  process_row(src, src, src + W, 0);

#pragma omp parallel for schedule(static)
  for (int y = 1; y < H - 1; ++y) {
    const float *p_up = src + (y - 1) * W;
    const float *p_mid = src + y * W;
    const float *p_dn = src + (y + 1) * W;

    float u0 = p_up[0], m0 = p_mid[0], d0 = p_dn[0];
    float u1 = p_up[0], m1 = p_mid[0], d1 = p_dn[0];
    float u2 = p_up[1], m2 = p_mid[1], d2 = p_dn[1];

    {
      float gx = (u2 + 2.0f * m2 + d2) - (u0 + 2.0f * m0 + d0);
      float gy = (d0 + 2.0f * d1 + d2) - (u0 + 2.0f * u1 + u2);
      p_mag[y * W] = __builtin_sqrtf(gx * gx + gy * gy);
      float abs_x = __builtin_fabsf(gx), abs_y = __builtin_fabsf(gy);
      uint8_t q = (abs_y <= abs_x * TAN_22_5)  ? 0
                  : (abs_y > abs_x * TAN_67_5) ? 90
                  : ((gx > 0) == (gy > 0))     ? 45
                                               : 135;
      p_dir[y * W] = q;
    }

    for (int x = 1; x < W - 1; ++x) {
      u0 = u1;
      m0 = m1;
      d0 = d1;
      u1 = u2;
      m1 = m2;
      d1 = d2;
      u2 = p_up[x + 1];
      m2 = p_mid[x + 1];
      d2 = p_dn[x + 1];

      float gx = (u2 + 2.0f * m2 + d2) - (u0 + 2.0f * m0 + d0);
      float gy = (d0 + 2.0f * d1 + d2) - (u0 + 2.0f * u1 + u2);
      int idx = y * W + x;
      p_mag[idx] = __builtin_sqrtf(gx * gx + gy * gy);
      float abs_x = __builtin_fabsf(gx), abs_y = __builtin_fabsf(gy);
      uint8_t q = (abs_y <= abs_x * TAN_22_5)  ? 0
                  : (abs_y > abs_x * TAN_67_5) ? 90
                  : ((gx > 0) == (gy > 0))     ? 45
                                               : 135;
      p_dir[idx] = q;
    }

    {
      u0 = u1;
      m0 = m1;
      d0 = d1;
      u1 = u2;
      m1 = m2;
      d1 = d2;
      u2 = u1;
      m2 = m1;
      d2 = d1;
      float gx = (u2 + 2 * m2 + d2) - (u0 + 2 * m0 + d0);
      float gy = (d0 + 2 * d1 + d2) - (u0 + 2 * u1 + u2);
      int idx = y * W + W - 1;
      p_mag[idx] = __builtin_sqrtf(gx * gx + gy * gy);
      float abs_x = __builtin_fabsf(gx), abs_y = __builtin_fabsf(gy);
      uint8_t q = (abs_y <= abs_x * TAN_22_5)  ? 0
                  : (abs_y > abs_x * TAN_67_5) ? 90
                  : ((gx > 0) == (gy > 0))     ? 45
                                               : 135;
      p_dir[idx] = q;
    }
  }

  process_row(src + (H - 2) * W, src + (H - 1) * W, src + (H - 1) * W, H - 1);
}

// ============================================================================
// NON-MAXIMUM SUPPRESSION: Branchless with LUT
// ============================================================================
static inline void process_nms_boundary(const float *mag, const uint8_t *dir,
                                        float *thin, int x, int y, int W,
                                        int H) {
  int idx = y * W + x;
  float m = mag[idx];
  uint8_t d = dir[idx];
  float m1 = 0, m2 = 0;
  auto clamp = [](int v, int min_v, int max_v) {
    return std::max(min_v, std::min(v, max_v));
  };

  if (d == 0) {
    m1 = mag[y * W + clamp(x - 1, 0, W - 1)];
    m2 = mag[y * W + clamp(x + 1, 0, W - 1)];
  } else if (d == 45) {
    m1 = mag[clamp(y - 1, 0, H - 1) * W + clamp(x + 1, 0, W - 1)];
    m2 = mag[clamp(y + 1, 0, H - 1) * W + clamp(x - 1, 0, W - 1)];
  } else if (d == 90) {
    m1 = mag[clamp(y - 1, 0, H - 1) * W + x];
    m2 = mag[clamp(y + 1, 0, H - 1) * W + x];
  } else {
    m1 = mag[clamp(y - 1, 0, H - 1) * W + clamp(x - 1, 0, W - 1)];
    m2 = mag[clamp(y + 1, 0, H - 1) * W + clamp(x + 1, 0, W - 1)];
  }
  thin[idx] = (m >= m1 && m >= m2) ? m : 0.0f;
}

static void nonmax_supp(const std::vector<float> &mag,
                        const std::vector<uint8_t> &dir,
                        std::vector<float> &thin, int W, int H) {
  const float *p_mag = mag.data();
  const uint8_t *p_dir = dir.data();
  float *p_thin = thin.data();

  const int offset1[4] = {-1, -W + 1, -W, -W - 1};
  const int offset2[4] = {+1, +W - 1, +W, +W + 1};

  // Boundaries - parallelize for 8k+ images
#pragma omp parallel for schedule(static) if (W >= 8000)
  for (int x = 0; x < W; ++x) {
    process_nms_boundary(p_mag, p_dir, p_thin, x, 0, W, H);
    process_nms_boundary(p_mag, p_dir, p_thin, x, H - 1, W, H);
  }
#pragma omp parallel for schedule(static) if (H >= 8000)
  for (int y = 1; y < H - 1; ++y) {
    process_nms_boundary(p_mag, p_dir, p_thin, 0, y, W, H);
    process_nms_boundary(p_mag, p_dir, p_thin, W - 1, y, W, H);
  }

  // Inner zone: branchless NMS
#pragma omp parallel for schedule(static)
  for (int y = 1; y < H - 1; ++y) {
    int row_base = y * W;
    for (int x = 1; x < W - 1; ++x) {
      int idx = row_base + x;
      float m = p_mag[idx];
      int d_idx = p_dir[idx] / 45;
      float m1 = p_mag[idx + offset1[d_idx]];
      float m2 = p_mag[idx + offset2[d_idx]];
      int keep = (m >= m1) & (m >= m2);
      p_thin[idx] = m * keep;
    }
  }
}

// ============================================================================
// DOUBLE THRESHOLD
// ============================================================================
static void double_threshold(const std::vector<float> &thin,
                             std::vector<uint8_t> &edges, float lowT,
                             float highT, int W, int H) {
  const int total = W * H;
  const float *src = thin.data();
  uint8_t *dst = edges.data();

#pragma omp parallel for schedule(static)
  for (int i = 0; i < total; ++i) {
    float v = src[i];
    dst[i] = (uint8_t)(v >= lowT) + (uint8_t)(v >= highT);
  }
}

// ============================================================================
// HYSTERESIS: Parallel Wavefront BFS
// ============================================================================
static void hysteresis(std::vector<uint8_t> &edges, int W, int H) {
  const int total = W * H;
  uint8_t *data = edges.data();

  std::vector<int> current_wave;
  current_wave.reserve(total / 100);

#pragma omp parallel
  {
    std::vector<int> local_seeds;
    local_seeds.reserve(total / (100 * omp_get_num_threads()));

#pragma omp for schedule(static) nowait
    for (int i = 0; i < total; ++i) {
      if (data[i] == 2) {
        data[i] = 255;
        local_seeds.push_back(i);
      }
    }

#pragma omp critical
    current_wave.insert(current_wave.end(), local_seeds.begin(),
                        local_seeds.end());
  }

  const int offsets[8] = {-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1};

  while (!current_wave.empty()) {
    std::vector<int> next_wave;
    next_wave.reserve(current_wave.size() * 2);

#pragma omp parallel
    {
      std::vector<int> local_next;
      local_next.reserve(current_wave.size() / omp_get_num_threads());

#pragma omp for schedule(static) nowait
      for (size_t i = 0; i < current_wave.size(); ++i) {
        int p = current_wave[i];
        int y = p / W, x = p % W;
        bool is_inner = (x > 0 && x < W - 1 && y > 0 && y < H - 1);

        if (is_inner) {
          for (int k = 0; k < 8; ++k) {
            int nidx = p + offsets[k];
            uint8_t expected = 1;
            if (__sync_bool_compare_and_swap(&data[nidx], expected, 255)) {
              local_next.push_back(nidx);
            }
          }
        } else {
          int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
          int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
          for (int k = 0; k < 8; ++k) {
            int nx = x + dx[k], ny = y + dy[k];
            if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
              int nidx = ny * W + nx;
              uint8_t expected = 1;
              if (__sync_bool_compare_and_swap(&data[nidx], expected, 255)) {
                local_next.push_back(nidx);
              }
            }
          }
        }
      }

#pragma omp critical
      next_wave.insert(next_wave.end(), local_next.begin(), local_next.end());
    }

    current_wave = std::move(next_wave);
  }

#pragma omp parallel for schedule(static)
  for (int i = 0; i < total; ++i) {
    if (data[i] != 255)
      data[i] = 0;
  }
}

// ============================================================================
// MAIN CANNY PIPELINE
// ============================================================================
static void canny(const std::vector<uint8_t> &in, std::vector<uint8_t> &out,
                  int W, int H) {
  const int total_pixels = W * H;

  std::vector<float> tmp(total_pixels), blur(total_pixels);
  std::vector<float> mag(total_pixels), thin(total_pixels);
  std::vector<uint8_t> dir(total_pixels), edges(total_pixels);

  gaussian5x1(in, tmp, W, H);
  gaussian1x5_from_tmp(tmp, blur, W, H);
  sobel_and_grad(blur, mag, dir, W, H);
  nonmax_supp(mag, dir, thin, W, H);

  // Auto threshold from thin values
  double sum = 0.0, sum2 = 0.0;
#pragma omp parallel for reduction(+ : sum, sum2)
  for (int i = 0; i < total_pixels; ++i) {
    float v = thin[i];
    sum += v;
    sum2 += v * v;
  }

  double mean = sum / total_pixels;
  double var = (sum2 / total_pixels) - (mean * mean);
  double stdv = __builtin_sqrt(__builtin_fmax(0.0, var));
  float highT = (float)(mean + stdv);
  float lowT = highT * 0.4f;

  double_threshold(thin, edges, lowT, highT, W, H);
  hysteresis(edges, W, H);

  out = std::move(edges);
}

} // namespace student