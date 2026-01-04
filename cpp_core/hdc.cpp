
#include "hdc.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>

EXPORT void bind_vectors(
    const int8_t* val_a,
    const int8_t* val_b,
    int8_t* result,
    int size
) {
    // #pragma omp parallel for // Potential for OpenMP if desired
    for (int i = 0; i < size; ++i) {
        // Bipolar Mult: 1*1=1, -1*-1=1, 1*-1=-1
        // Same as XOR in binary space (0,1).
        result[i] = val_a[i] * val_b[i];
    }
}

EXPORT void bundle_vectors(
    const int8_t* inputs_flat,
    int num_vectors,
    int8_t* result,
    int size
) {
    // We need a temporary integer buffer for summation
    std::vector<int> sum_vec(size, 0);

    for (int n = 0; n < num_vectors; ++n) {
        const int8_t* vec = inputs_flat + (n * size);
        for (int i = 0; i < size; ++i) {
            sum_vec[i] += vec[i];
        }
    }

    // Threshold (Majority Rule)
    for (int i = 0; i < size; ++i) {
        if (sum_vec[i] > 0) result[i] = 1;
        else if (sum_vec[i] < 0) result[i] = -1;
        else {
            // Random tie break
            result[i] = (rand() % 2 == 0) ? 1 : -1;
        }
    }
}

EXPORT double cosine_similarity(
    const int8_t* val_a,
    const int8_t* val_b,
    int size
) {
    long long dot_product = 0;
    for (int i = 0; i < size; ++i) {
        dot_product += val_a[i] * val_b[i];
    }
    // Normalized [-1, 1]
    return (double)dot_product / (double)size;
}
