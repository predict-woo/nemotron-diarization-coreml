// Top-k selection with the exact tie-breaking of PyTorch's CPU kernel (aten/src/ATen/native/TopKImpl.h):
// (value, index) pairs, std::nth_element with a "greater" comparator (NaN largest), first k kept.
// Equal values are therefore resolved by libc++'s introselect, not by index order.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

extern "C" void torch_topk_largest(const float* values, int64_t n, int64_t k, int64_t* out_indices) {
    using elem_t = std::pair<double, int64_t>;  // accscalar_t of float on CPU is double
    std::vector<elem_t> queue(n);
    for (int64_t j = 0; j < n; ++j) queue[j] = {static_cast<double>(values[j]), j};
    auto greater = [](const elem_t& x, const elem_t& y) -> bool {
        return (std::isnan(x.first) && !std::isnan(y.first)) || (x.first > y.first);
    };
    if (k * 64 <= n) {
        std::partial_sort(queue.begin(), queue.begin() + k, queue.end(), greater);
    } else {
        std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(), greater);
    }
    for (int64_t j = 0; j < k; ++j) out_indices[j] = queue[j].second;
}
