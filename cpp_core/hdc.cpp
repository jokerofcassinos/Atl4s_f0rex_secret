
#include "hdc.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <random>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// GLOBAL STATE
// ============================================================================

static const char* HDC_VERSION = "2.0.0-AGI-Ultra";
static thread_local std::mt19937 tl_gen(std::random_device{}());

// ============================================================================
// ORIGINAL HDC OPERATIONS (Backward Compatible)
// ============================================================================

EXPORT void bind_vectors(
    const int8_t* val_a,
    const int8_t* val_b,
    int8_t* result,
    int size
) {
    #pragma omp parallel for if(size > 1000)
    for (int i = 0; i < size; ++i) {
        result[i] = val_a[i] * val_b[i];
    }
}

EXPORT void bundle_vectors(
    const int8_t* inputs_flat,
    int num_vectors,
    int8_t* result,
    int size
) {
    std::vector<int> sum_vec(size, 0);

    for (int n = 0; n < num_vectors; ++n) {
        const int8_t* vec = inputs_flat + (n * size);
        for (int i = 0; i < size; ++i) {
            sum_vec[i] += vec[i];
        }
    }

    for (int i = 0; i < size; ++i) {
        if (sum_vec[i] > 0) result[i] = 1;
        else if (sum_vec[i] < 0) result[i] = -1;
        else {
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
    #pragma omp parallel for reduction(+:dot_product) if(size > 1000)
    for (int i = 0; i < size; ++i) {
        dot_product += val_a[i] * val_b[i];
    }
    return (double)dot_product / (double)size;
}

// ============================================================================
// SPARSE HDC
// ============================================================================

EXPORT SparseVector* create_sparse_vector(
    const int8_t* dense,
    int size,
    double sparsity_threshold
) {
    SparseVector* vec = new SparseVector();
    vec->dimension = size;
    
    // Count non-zeros
    std::vector<int> indices;
    std::vector<int8_t> values;
    
    for (int i = 0; i < size; i++) {
        if (dense[i] != 0) {
            indices.push_back(i);
            values.push_back(dense[i]);
        }
    }
    
    vec->nnz = indices.size();
    vec->indices = new int[vec->nnz];
    vec->values = new int8_t[vec->nnz];
    
    std::copy(indices.begin(), indices.end(), vec->indices);
    std::copy(values.begin(), values.end(), vec->values);
    
    return vec;
}

EXPORT void free_sparse_vector(SparseVector* vec) {
    if (vec) {
        delete[] vec->indices;
        delete[] vec->values;
        delete vec;
    }
}

EXPORT void bind_sparse_vectors(
    const int* indices_a, const int8_t* values_a, int size_a,
    const int* indices_b, const int8_t* values_b, int size_b,
    int8_t* result, int result_size
) {
    std::memset(result, 0, result_size);
    
    // Create index map for vector b
    std::unordered_set<int> b_indices(indices_b, indices_b + size_b);
    
    // Only non-zero where both are non-zero
    for (int i = 0; i < size_a; i++) {
        int idx = indices_a[i];
        if (b_indices.count(idx) > 0) {
            // Find value in b
            for (int j = 0; j < size_b; j++) {
                if (indices_b[j] == idx) {
                    result[idx] = values_a[i] * values_b[j];
                    break;
                }
            }
        }
    }
}

EXPORT void bundle_sparse_vectors(
    SparseVector** vectors,
    int num_vectors,
    int8_t* result,
    int result_size
) {
    std::vector<int> sum_vec(result_size, 0);
    
    for (int v = 0; v < num_vectors; v++) {
        SparseVector* vec = vectors[v];
        for (int i = 0; i < vec->nnz; i++) {
            sum_vec[vec->indices[i]] += vec->values[i];
        }
    }
    
    for (int i = 0; i < result_size; i++) {
        if (sum_vec[i] > 0) result[i] = 1;
        else if (sum_vec[i] < 0) result[i] = -1;
        else result[i] = (rand() % 2 == 0) ? 1 : -1;
    }
}

EXPORT double sparse_cosine_similarity(
    const SparseVector* a,
    const SparseVector* b
) {
    if (!a || !b || a->dimension != b->dimension) return 0.0;
    
    long long dot = 0;
    
    // O(nnz_a * nnz_b) - could be optimized with sorted indices
    for (int i = 0; i < a->nnz; i++) {
        for (int j = 0; j < b->nnz; j++) {
            if (a->indices[i] == b->indices[j]) {
                dot += a->values[i] * b->values[j];
                break;
            }
        }
    }
    
    return (double)dot / (double)a->dimension;
}

// ============================================================================
// QUANTIZED HDC
// ============================================================================

EXPORT void quantize_vector(
    const float* input,
    int8_t* output,
    int size,
    float scale
) {
    #pragma omp parallel for if(size > 1000)
    for (int i = 0; i < size; i++) {
        float val = input[i] * scale;
        if (val > 127) val = 127;
        if (val < -127) val = -127;
        output[i] = (int8_t)val;
    }
}

EXPORT void dequantize_vector(
    const int8_t* input,
    float* output,
    int size,
    float scale
) {
    #pragma omp parallel for if(size > 1000)
    for (int i = 0; i < size; i++) {
        output[i] = (float)input[i] / scale;
    }
}

EXPORT void bind_quantized(
    const int8_t* a, float scale_a,
    const int8_t* b, float scale_b,
    int8_t* result, float* result_scale,
    int size
) {
    *result_scale = scale_a * scale_b;
    
    #pragma omp parallel for if(size > 1000)
    for (int i = 0; i < size; i++) {
        int val = (int)a[i] * (int)b[i];
        if (val > 127) val = 127;
        if (val < -127) val = -127;
        result[i] = (int8_t)val;
    }
}

EXPORT void bundle_quantized(
    const int8_t* inputs_flat,
    const float* scales,
    int num_vectors,
    int8_t* result,
    float* result_scale,
    int size
) {
    std::vector<float> sum_vec(size, 0.0f);
    
    for (int n = 0; n < num_vectors; n++) {
        const int8_t* vec = inputs_flat + (n * size);
        float scale = scales[n];
        
        for (int i = 0; i < size; i++) {
            sum_vec[i] += ((float)vec[i] / scale);
        }
    }
    
    // Find max for scaling
    float max_val = 0.0f;
    for (int i = 0; i < size; i++) {
        if (std::abs(sum_vec[i]) > max_val) {
            max_val = std::abs(sum_vec[i]);
        }
    }
    
    *result_scale = (max_val > 0) ? 127.0f / max_val : 1.0f;
    
    for (int i = 0; i < size; i++) {
        float val = sum_vec[i] * (*result_scale);
        if (val > 127) val = 127;
        if (val < -127) val = -127;
        result[i] = (int8_t)val;
    }
}

// ============================================================================
// HIERARCHICAL HDC
// ============================================================================

EXPORT HierarchicalHDC* create_hierarchical_hdc(
    const int8_t* base_vector,
    int dimension,
    int num_levels
) {
    HierarchicalHDC* hdc = new HierarchicalHDC();
    hdc->dimension = dimension;
    hdc->num_levels = num_levels;
    hdc->levels = new int8_t*[num_levels];
    
    // Level 0 is the base
    hdc->levels[0] = new int8_t[dimension];
    std::memcpy(hdc->levels[0], base_vector, dimension);
    
    // Higher levels are permuted versions
    for (int l = 1; l < num_levels; l++) {
        hdc->levels[l] = new int8_t[dimension];
        int shift = l * (dimension / num_levels);
        
        for (int i = 0; i < dimension; i++) {
            int src = (i + shift) % dimension;
            hdc->levels[l][i] = hdc->levels[l-1][src];
        }
    }
    
    return hdc;
}

EXPORT void free_hierarchical_hdc(HierarchicalHDC* hdc) {
    if (hdc) {
        for (int l = 0; l < hdc->num_levels; l++) {
            delete[] hdc->levels[l];
        }
        delete[] hdc->levels;
        delete hdc;
    }
}

EXPORT void create_hierarchical_bundle(
    const int8_t* level1,
    const int8_t* level2,
    const int8_t* level3,
    int8_t* result,
    int size
) {
    std::vector<int> sum_vec(size, 0);
    
    // Weight levels differently (higher levels get more weight)
    for (int i = 0; i < size; i++) {
        sum_vec[i] = level1[i] + 2 * level2[i] + 3 * level3[i];
    }
    
    for (int i = 0; i < size; i++) {
        if (sum_vec[i] > 0) result[i] = 1;
        else if (sum_vec[i] < 0) result[i] = -1;
        else result[i] = 1;
    }
}

EXPORT double hierarchical_similarity(
    const HierarchicalHDC* hdc,
    const int8_t* query,
    int level
) {
    if (!hdc || level < 0 || level >= hdc->num_levels) return 0.0;
    return cosine_similarity(hdc->levels[level], query, hdc->dimension);
}

// ============================================================================
// TEMPORAL HDC
// ============================================================================

EXPORT TemporalHDC* create_temporal_hdc(
    const int8_t* sequence_flat,
    int seq_length,
    int dimension
) {
    TemporalHDC* hdc = new TemporalHDC();
    hdc->seq_length = seq_length;
    hdc->dimension = dimension;
    
    hdc->sequence = new int8_t*[seq_length];
    for (int t = 0; t < seq_length; t++) {
        hdc->sequence[t] = new int8_t[dimension];
        std::memcpy(hdc->sequence[t], sequence_flat + t * dimension, dimension);
    }
    
    // Create position encoding vectors
    hdc->position_vectors = new int8_t[seq_length * dimension];
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<> dist(0, 1);
    
    for (int t = 0; t < seq_length; t++) {
        for (int d = 0; d < dimension; d++) {
            hdc->position_vectors[t * dimension + d] = dist(gen) ? 1 : -1;
        }
    }
    
    return hdc;
}

EXPORT void free_temporal_hdc(TemporalHDC* hdc) {
    if (hdc) {
        for (int t = 0; t < hdc->seq_length; t++) {
            delete[] hdc->sequence[t];
        }
        delete[] hdc->sequence;
        delete[] hdc->position_vectors;
        delete hdc;
    }
}

EXPORT void create_temporal_sequence(
    const int8_t* sequence_flat,
    int sequence_length,
    int8_t* result,
    int size
) {
    std::vector<int> sum_vec(size, 0);
    
    for (int t = 0; t < sequence_length; t++) {
        const int8_t* vec = sequence_flat + t * size;
        int shift = t;  // Temporal shift
        
        for (int i = 0; i < size; i++) {
            int idx = (i + shift) % size;
            sum_vec[i] += vec[idx];
        }
    }
    
    for (int i = 0; i < size; i++) {
        if (sum_vec[i] > 0) result[i] = 1;
        else if (sum_vec[i] < 0) result[i] = -1;
        else result[i] = 1;
    }
}

EXPORT void create_ngram_encoding(
    const int8_t* sequence_flat,
    int sequence_length,
    int n,
    int8_t* result,
    int size
) {
    std::vector<int> sum_vec(size, 0);
    
    for (int t = 0; t <= sequence_length - n; t++) {
        // Bind n consecutive vectors
        std::vector<int8_t> ngram(size);
        std::memcpy(ngram.data(), sequence_flat + t * size, size);
        
        for (int k = 1; k < n; k++) {
            const int8_t* next = sequence_flat + (t + k) * size;
            int shift = k;
            
            for (int i = 0; i < size; i++) {
                int idx = (i + shift) % size;
                ngram[i] *= next[idx];
            }
        }
        
        // Add to bundle
        for (int i = 0; i < size; i++) {
            sum_vec[i] += ngram[i];
        }
    }
    
    for (int i = 0; i < size; i++) {
        if (sum_vec[i] > 0) result[i] = 1;
        else if (sum_vec[i] < 0) result[i] = -1;
        else result[i] = 1;
    }
}

// ============================================================================
// MULTI-MODAL HDC
// ============================================================================

EXPORT void fuse_modalities(
    const int8_t* price_vector,
    const int8_t* volume_vector,
    const int8_t* time_vector,
    int8_t* result,
    int size
) {
    std::vector<int> sum_vec(size, 0);
    
    // Different weights for modalities
    for (int i = 0; i < size; i++) {
        sum_vec[i] = 3 * price_vector[i] + 2 * volume_vector[i] + time_vector[i];
    }
    
    for (int i = 0; i < size; i++) {
        if (sum_vec[i] > 0) result[i] = 1;
        else if (sum_vec[i] < 0) result[i] = -1;
        else result[i] = 1;
    }
}

EXPORT void fuse_weighted_modalities(
    const int8_t** modalities,
    const double* weights,
    int num_modalities,
    int8_t* result,
    int size
) {
    std::vector<double> sum_vec(size, 0.0);
    
    for (int m = 0; m < num_modalities; m++) {
        for (int i = 0; i < size; i++) {
            sum_vec[i] += weights[m] * modalities[m][i];
        }
    }
    
    for (int i = 0; i < size; i++) {
        if (sum_vec[i] > 0) result[i] = 1;
        else if (sum_vec[i] < 0) result[i] = -1;
        else result[i] = 1;
    }
}

// ============================================================================
// HDC CLASSIFIER
// ============================================================================

EXPORT HDCClassifier* create_hdc_classifier(int num_classes, int dimension) {
    HDCClassifier* clf = new HDCClassifier();
    clf->num_classes = num_classes;
    clf->dimension = dimension;
    
    clf->class_vectors = new int8_t*[num_classes];
    clf->class_counts = new int[num_classes];
    
    for (int c = 0; c < num_classes; c++) {
        clf->class_vectors[c] = new int8_t[dimension];
        std::memset(clf->class_vectors[c], 0, dimension);
        clf->class_counts[c] = 0;
    }
    
    return clf;
}

EXPORT void free_hdc_classifier(HDCClassifier* clf) {
    if (clf) {
        for (int c = 0; c < clf->num_classes; c++) {
            delete[] clf->class_vectors[c];
        }
        delete[] clf->class_vectors;
        delete[] clf->class_counts;
        delete clf;
    }
}

EXPORT void train_hdc_classifier(
    HDCClassifier* clf,
    const int8_t* training_vectors,
    const int* labels,
    int num_samples
) {
    // Accumulate vectors per class
    std::vector<std::vector<int>> sums(clf->num_classes, std::vector<int>(clf->dimension, 0));
    
    for (int s = 0; s < num_samples; s++) {
        int label = labels[s];
        if (label < 0 || label >= clf->num_classes) continue;
        
        const int8_t* vec = training_vectors + s * clf->dimension;
        for (int d = 0; d < clf->dimension; d++) {
            sums[label][d] += vec[d];
        }
        clf->class_counts[label]++;
    }
    
    // Threshold to bipolar
    for (int c = 0; c < clf->num_classes; c++) {
        for (int d = 0; d < clf->dimension; d++) {
            if (sums[c][d] > 0) clf->class_vectors[c][d] = 1;
            else if (sums[c][d] < 0) clf->class_vectors[c][d] = -1;
            else clf->class_vectors[c][d] = 1;
        }
    }
}

EXPORT int predict_hdc(const HDCClassifier* clf, const int8_t* query) {
    double max_sim = -2.0;
    int best_class = 0;
    
    for (int c = 0; c < clf->num_classes; c++) {
        double sim = cosine_similarity(clf->class_vectors[c], query, clf->dimension);
        if (sim > max_sim) {
            max_sim = sim;
            best_class = c;
        }
    }
    
    return best_class;
}

EXPORT int predict_hdc_with_confidence(
    const HDCClassifier* clf,
    const int8_t* query,
    double* confidence
) {
    double max_sim = -2.0;
    double second_sim = -2.0;
    int best_class = 0;
    
    for (int c = 0; c < clf->num_classes; c++) {
        double sim = cosine_similarity(clf->class_vectors[c], query, clf->dimension);
        if (sim > max_sim) {
            second_sim = max_sim;
            max_sim = sim;
            best_class = c;
        } else if (sim > second_sim) {
            second_sim = sim;
        }
    }
    
    // Confidence based on margin
    *confidence = (max_sim - second_sim) / 2.0 + 0.5;
    *confidence = std::max(0.0, std::min(1.0, *confidence));
    
    return best_class;
}

// ============================================================================
// HDC MEMORY
// ============================================================================

EXPORT HDCMemory* create_hdc_memory(int max_items, int dimension) {
    HDCMemory* mem = new HDCMemory();
    mem->max_items = max_items;
    mem->dimension = dimension;
    mem->num_items = 0;
    
    mem->keys = new int8_t*[max_items];
    mem->values = new int8_t*[max_items];
    
    for (int i = 0; i < max_items; i++) {
        mem->keys[i] = nullptr;
        mem->values[i] = nullptr;
    }
    
    return mem;
}

EXPORT void free_hdc_memory(HDCMemory* mem) {
    if (mem) {
        for (int i = 0; i < mem->num_items; i++) {
            if (mem->keys[i]) delete[] mem->keys[i];
            if (mem->values[i]) delete[] mem->values[i];
        }
        delete[] mem->keys;
        delete[] mem->values;
        delete mem;
    }
}

EXPORT void hdc_memory_store(HDCMemory* mem, const int8_t* key, const int8_t* value) {
    if (mem->num_items >= mem->max_items) return;
    
    mem->keys[mem->num_items] = new int8_t[mem->dimension];
    mem->values[mem->num_items] = new int8_t[mem->dimension];
    
    std::memcpy(mem->keys[mem->num_items], key, mem->dimension);
    std::memcpy(mem->values[mem->num_items], value, mem->dimension);
    
    mem->num_items++;
}

EXPORT int hdc_memory_retrieve(
    const HDCMemory* mem,
    const int8_t* query,
    int8_t* result,
    double threshold
) {
    double max_sim = -2.0;
    int best_idx = -1;
    
    for (int i = 0; i < mem->num_items; i++) {
        double sim = cosine_similarity(mem->keys[i], query, mem->dimension);
        if (sim > max_sim) {
            max_sim = sim;
            best_idx = i;
        }
    }
    
    if (best_idx >= 0 && max_sim >= threshold) {
        std::memcpy(result, mem->values[best_idx], mem->dimension);
        return 1;
    }
    
    return 0;
}

EXPORT void hdc_memory_cleanup(HDCMemory* mem, double similarity_threshold) {
    std::vector<bool> keep(mem->num_items, true);
    
    // Mark similar items for removal
    for (int i = 0; i < mem->num_items; i++) {
        if (!keep[i]) continue;
        
        for (int j = i + 1; j < mem->num_items; j++) {
            if (!keep[j]) continue;
            
            double sim = cosine_similarity(mem->keys[i], mem->keys[j], mem->dimension);
            if (sim > similarity_threshold) {
                keep[j] = false;
            }
        }
    }
    
    // Compact
    int write_idx = 0;
    for (int i = 0; i < mem->num_items; i++) {
        if (keep[i]) {
            if (write_idx != i) {
                mem->keys[write_idx] = mem->keys[i];
                mem->values[write_idx] = mem->values[i];
            }
            write_idx++;
        } else {
            delete[] mem->keys[i];
            delete[] mem->values[i];
        }
    }
    
    mem->num_items = write_idx;
}

// ============================================================================
// ENCODING FUNCTIONS
// ============================================================================

EXPORT void encode_scalar(
    double value,
    double min_val,
    double max_val,
    int8_t* result,
    int dimension
) {
    // Level encoding
    double normalized = (value - min_val) / (max_val - min_val + 0.001);
    int level = (int)(normalized * dimension);
    if (level < 0) level = 0;
    if (level >= dimension) level = dimension - 1;
    
    // Thermometer encoding
    for (int i = 0; i < dimension; i++) {
        result[i] = (i <= level) ? 1 : -1;
    }
}

EXPORT void encode_time_series(
    const double* values,
    int length,
    int8_t* result,
    int dimension
) {
    // Find min/max
    double min_val = values[0], max_val = values[0];
    for (int i = 1; i < length; i++) {
        if (values[i] < min_val) min_val = values[i];
        if (values[i] > max_val) max_val = values[i];
    }
    
    // Encode each value and bundle
    std::vector<int> sum_vec(dimension, 0);
    std::vector<int8_t> encoded(dimension);
    
    for (int t = 0; t < length; t++) {
        encode_scalar(values[t], min_val, max_val, encoded.data(), dimension);
        
        // Temporal shift
        for (int i = 0; i < dimension; i++) {
            int idx = (i + t) % dimension;
            sum_vec[i] += encoded[idx];
        }
    }
    
    for (int i = 0; i < dimension; i++) {
        if (sum_vec[i] > 0) result[i] = 1;
        else if (sum_vec[i] < 0) result[i] = -1;
        else result[i] = 1;
    }
}

EXPORT void encode_categorical(
    int category,
    int num_categories,
    int8_t* result,
    int dimension
) {
    // Use deterministic random based on category
    std::mt19937 gen(category * 12345);
    std::uniform_int_distribution<> dist(0, 1);
    
    for (int i = 0; i < dimension; i++) {
        result[i] = dist(gen) ? 1 : -1;
    }
}

// ============================================================================
// GPU ACCELERATION (CPU Fallback)
// ============================================================================

EXPORT int hdc_gpu_available() {
    return 0;  // No GPU support in this version
}

EXPORT void bind_vectors_gpu(
    const int8_t* val_a,
    const int8_t* val_b,
    int8_t* result,
    int size
) {
    bind_vectors(val_a, val_b, result, size);  // Fallback to CPU
}

EXPORT void bundle_vectors_gpu(
    const int8_t* inputs_flat,
    int num_vectors,
    int8_t* result,
    int size
) {
    bundle_vectors(inputs_flat, num_vectors, result, size);  // Fallback to CPU
}

EXPORT void batch_similarity_gpu(
    const int8_t* queries,
    const int8_t* database,
    int num_queries,
    int num_db,
    int dimension,
    double* results
) {
    #pragma omp parallel for
    for (int q = 0; q < num_queries; q++) {
        const int8_t* query = queries + q * dimension;
        for (int d = 0; d < num_db; d++) {
            const int8_t* db_vec = database + d * dimension;
            results[q * num_db + d] = cosine_similarity(query, db_vec, dimension);
        }
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

EXPORT void generate_random_hv(int8_t* result, int dimension, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dist(0, 1);
    
    for (int i = 0; i < dimension; i++) {
        result[i] = dist(gen) ? 1 : -1;
    }
}

EXPORT void permute_vector(
    const int8_t* input,
    int8_t* output,
    int size,
    int shift
) {
    shift = shift % size;
    if (shift < 0) shift += size;
    
    for (int i = 0; i < size; i++) {
        int src = (i + shift) % size;
        output[i] = input[src];
    }
}

EXPORT const char* get_hdc_version() {
    return HDC_VERSION;
}
