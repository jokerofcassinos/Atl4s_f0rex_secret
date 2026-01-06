
#ifndef HDC_H
#define HDC_H

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

#include <cstdint>

extern "C" {
    // ============================================================================
    // HDC ULTRA-ADVANCED - AGI Core
    // Features: Sparse HDC, Quantized HDC, Hierarchical HDC, Temporal HDC
    // ============================================================================

    // ============================================================================
    // ORIGINAL HDC OPERATIONS (Backward Compatible)
    // ============================================================================

    // Bind (XOR) two dense bipolar vectors (-1, 1)
    EXPORT void bind_vectors(
        const int8_t* val_a,
        const int8_t* val_b,
        int8_t* result,
        int size
    );

    // Bundle (Add) multiple vectors
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

    // ============================================================================
    // SPARSE HDC
    // ============================================================================
    
    struct SparseVector {
        int* indices;           // Non-zero indices
        int8_t* values;         // Values at those indices
        int nnz;                // Number of non-zeros
        int dimension;          // Full dimension
    };
    
    // Create sparse vector from dense
    EXPORT SparseVector* create_sparse_vector(
        const int8_t* dense,
        int size,
        double sparsity_threshold
    );
    
    // Free sparse vector
    EXPORT void free_sparse_vector(SparseVector* vec);
    
    // Sparse bind
    EXPORT void bind_sparse_vectors(
        const int* indices_a, const int8_t* values_a, int size_a,
        const int* indices_b, const int8_t* values_b, int size_b,
        int8_t* result, int result_size
    );
    
    // Sparse bundle
    EXPORT void bundle_sparse_vectors(
        SparseVector** vectors,
        int num_vectors,
        int8_t* result,
        int result_size
    );
    
    // Sparse similarity
    EXPORT double sparse_cosine_similarity(
        const SparseVector* a,
        const SparseVector* b
    );

    // ============================================================================
    // QUANTIZED HDC
    // ============================================================================
    
    // Quantize float vector to int8
    EXPORT void quantize_vector(
        const float* input,
        int8_t* output,
        int size,
        float scale
    );
    
    // Dequantize int8 vector to float
    EXPORT void dequantize_vector(
        const int8_t* input,
        float* output,
        int size,
        float scale
    );
    
    // Quantized bind (with scale)
    EXPORT void bind_quantized(
        const int8_t* a, float scale_a,
        const int8_t* b, float scale_b,
        int8_t* result, float* result_scale,
        int size
    );
    
    // Quantized bundle with averaging
    EXPORT void bundle_quantized(
        const int8_t* inputs_flat,
        const float* scales,
        int num_vectors,
        int8_t* result,
        float* result_scale,
        int size
    );

    // ============================================================================
    // HIERARCHICAL HDC
    // ============================================================================
    
    struct HierarchicalHDC {
        int8_t** levels;        // Vectors at each level
        int num_levels;
        int dimension;
    };
    
    // Create hierarchical encoding
    EXPORT HierarchicalHDC* create_hierarchical_hdc(
        const int8_t* base_vector,
        int dimension,
        int num_levels
    );
    
    // Free hierarchical HDC
    EXPORT void free_hierarchical_hdc(HierarchicalHDC* hdc);
    
    // Bundle hierarchical levels
    EXPORT void create_hierarchical_bundle(
        const int8_t* level1,
        const int8_t* level2,
        const int8_t* level3,
        int8_t* result,
        int size
    );
    
    // Query at specific level
    EXPORT double hierarchical_similarity(
        const HierarchicalHDC* hdc,
        const int8_t* query,
        int level
    );

    // ============================================================================
    // TEMPORAL HDC
    // ============================================================================
    
    struct TemporalHDC {
        int8_t** sequence;      // Sequence of vectors
        int8_t* position_vectors; // Position encoding vectors
        int seq_length;
        int dimension;
    };
    
    // Create temporal encoding
    EXPORT TemporalHDC* create_temporal_hdc(
        const int8_t* sequence_flat,
        int seq_length,
        int dimension
    );
    
    // Free temporal HDC
    EXPORT void free_temporal_hdc(TemporalHDC* hdc);
    
    // Create temporal sequence encoding
    EXPORT void create_temporal_sequence(
        const int8_t* sequence_flat,
        int sequence_length,
        int8_t* result,
        int size
    );
    
    // N-gram encoding
    EXPORT void create_ngram_encoding(
        const int8_t* sequence_flat,
        int sequence_length,
        int n,
        int8_t* result,
        int size
    );

    // ============================================================================
    // MULTI-MODAL HDC
    // ============================================================================
    
    // Fuse multiple modalities with weights
    EXPORT void fuse_modalities(
        const int8_t* price_vector,
        const int8_t* volume_vector,
        const int8_t* time_vector,
        int8_t* result,
        int size
    );
    
    // Weighted modality fusion
    EXPORT void fuse_weighted_modalities(
        const int8_t** modalities,
        const double* weights,
        int num_modalities,
        int8_t* result,
        int size
    );

    // ============================================================================
    // HDC CLASSIFIER
    // ============================================================================
    
    struct HDCClassifier {
        int8_t** class_vectors;  // Prototype vectors for each class
        int num_classes;
        int dimension;
        int* class_counts;       // Training samples per class
    };
    
    // Create classifier
    EXPORT HDCClassifier* create_hdc_classifier(
        int num_classes,
        int dimension
    );
    
    // Free classifier
    EXPORT void free_hdc_classifier(HDCClassifier* clf);
    
    // Train classifier
    EXPORT void train_hdc_classifier(
        HDCClassifier* clf,
        const int8_t* training_vectors,
        const int* labels,
        int num_samples
    );
    
    // Predict class
    EXPORT int predict_hdc(
        const HDCClassifier* clf,
        const int8_t* query
    );
    
    // Predict with confidence
    EXPORT int predict_hdc_with_confidence(
        const HDCClassifier* clf,
        const int8_t* query,
        double* confidence
    );

    // ============================================================================
    // HDC MEMORY (Associative Memory)
    // ============================================================================
    
    struct HDCMemory {
        int8_t** keys;
        int8_t** values;
        int num_items;
        int max_items;
        int dimension;
    };
    
    // Create associative memory
    EXPORT HDCMemory* create_hdc_memory(
        int max_items,
        int dimension
    );
    
    // Free memory
    EXPORT void free_hdc_memory(HDCMemory* mem);
    
    // Store key-value pair
    EXPORT void hdc_memory_store(
        HDCMemory* mem,
        const int8_t* key,
        const int8_t* value
    );
    
    // Retrieve value by key
    EXPORT int hdc_memory_retrieve(
        const HDCMemory* mem,
        const int8_t* query,
        int8_t* result,
        double threshold
    );
    
    // Cleanup/compress memory
    EXPORT void hdc_memory_cleanup(
        HDCMemory* mem,
        double similarity_threshold
    );

    // ============================================================================
    // ENCODING FUNCTIONS
    // ============================================================================
    
    // Encode scalar value
    EXPORT void encode_scalar(
        double value,
        double min_val,
        double max_val,
        int8_t* result,
        int dimension
    );
    
    // Encode time series
    EXPORT void encode_time_series(
        const double* values,
        int length,
        int8_t* result,
        int dimension
    );
    
    // Encode categorical value
    EXPORT void encode_categorical(
        int category,
        int num_categories,
        int8_t* result,
        int dimension
    );

    // ============================================================================
    // GPU ACCELERATION (Placeholders for CUDA/OpenCL)
    // ============================================================================
    
    // Check GPU availability
    EXPORT int hdc_gpu_available();
    
    // GPU-accelerated bind
    EXPORT void bind_vectors_gpu(
        const int8_t* val_a,
        const int8_t* val_b,
        int8_t* result,
        int size
    );
    
    // GPU-accelerated bundle
    EXPORT void bundle_vectors_gpu(
        const int8_t* inputs_flat,
        int num_vectors,
        int8_t* result,
        int size
    );
    
    // GPU-accelerated batch similarity
    EXPORT void batch_similarity_gpu(
        const int8_t* queries,
        const int8_t* database,
        int num_queries,
        int num_db,
        int dimension,
        double* results
    );

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================
    
    // Generate random hypervector
    EXPORT void generate_random_hv(
        int8_t* result,
        int dimension,
        unsigned int seed
    );
    
    // Permute vector (circular shift)
    EXPORT void permute_vector(
        const int8_t* input,
        int8_t* output,
        int size,
        int shift
    );
    
    // Get version
    EXPORT const char* get_hdc_version();
}

#endif
