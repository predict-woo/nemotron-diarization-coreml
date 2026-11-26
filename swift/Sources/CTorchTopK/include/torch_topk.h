#ifndef TORCH_TOPK_H
#define TORCH_TOPK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Indices of the `k` largest of `values[0..<n]`, in the order and with the tie-breaking of `torch.topk(...,
/// largest=True, sorted=False)` on CPU: (value, index) pairs partitioned by std::nth_element (std::partial_sort
/// when k * 64 <= n).
void torch_topk_largest(const float *values, int64_t n, int64_t k, int64_t *out_indices);

#ifdef __cplusplus
}
#endif

#endif
