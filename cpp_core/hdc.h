
#ifndef HDC_H
#define HDC_H

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

#include <cstdint>

extern "C" {
    // Bind (XOR) two dense bipolar vectors (-1, 1).
    // result = a * b
    // size: Dimension (D)
    EXPORT void bind_vectors(
        const int8_t* val_a,
        const int8_t* val_b,
        int8_t* result,
        int size
    );

    // Bundle (Add) multiple vectors
    // inputs: flattened array of N vectors
    // result: bipolar vector (-1, 1)
    EXPORT void bundle_vectors(
        const int8_t* inputs_flat,
        int num_vectors,
        int8_t* result,
        int size
    );

    // Cosine Similarity
    EXPORT double cosine_similarity(
        const int8_t* val_a,
        const int8_t* val_b,
        int size
    );
}

#endif
