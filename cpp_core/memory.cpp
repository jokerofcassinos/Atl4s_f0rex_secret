
#include "memory.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <queue>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// GLOBAL STATE
// ============================================================================

static const char* MEMORY_VERSION = "1.0.0-AGI";
static thread_local std::mt19937 tl_gen(std::random_device{}());

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

EXPORT float l2_distance(const float* a, const float* b, int dimension) {
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < dimension; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

EXPORT float cosine_similarity_f(const float* a, const float* b, int dimension) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    
    #pragma omp simd reduction(+:dot,norm_a,norm_b)
    for (int i = 0; i < dimension; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-10f);
}

EXPORT float dot_product(const float* a, const float* b, int dimension) {
    float dot = 0.0f;
    #pragma omp simd reduction(+:dot)
    for (int i = 0; i < dimension; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

EXPORT void normalize_vector(float* vector, int dimension) {
    float norm = 0.0f;
    for (int i = 0; i < dimension; i++) {
        norm += vector[i] * vector[i];
    }
    norm = std::sqrt(norm) + 1e-10f;
    
    for (int i = 0; i < dimension; i++) {
        vector[i] /= norm;
    }
}

// ============================================================================
// VECTOR DATABASE
// ============================================================================

EXPORT VectorDB* create_vector_db(int dimension, size_t max_vectors) {
    VectorDB* db = new VectorDB();
    db->dimension = dimension;
    db->max_entries = max_vectors;
    db->num_entries = 0;
    db->entries = new VectorEntry[max_vectors];
    db->normalized_cache = new float[max_vectors * dimension];
    
    for (size_t i = 0; i < max_vectors; i++) {
        db->entries[i].vector = nullptr;
    }
    
    return db;
}

EXPORT void free_vector_db(VectorDB* db) {
    if (db) {
        for (size_t i = 0; i < db->num_entries; i++) {
            delete[] db->entries[i].vector;
        }
        delete[] db->entries;
        delete[] db->normalized_cache;
        delete db;
    }
}

EXPORT size_t add_vector(VectorDB* db, const float* vector, const char* metadata) {
    if (db->num_entries >= db->max_entries) return (size_t)-1;
    
    VectorEntry* entry = &db->entries[db->num_entries];
    entry->id = db->num_entries;
    entry->dimension = db->dimension;
    entry->vector = new float[db->dimension];
    memcpy(entry->vector, vector, db->dimension * sizeof(float));
    entry->timestamp = 0;
    entry->importance = 1.0;
    
    if (metadata) {
        strncpy(entry->metadata, metadata, sizeof(entry->metadata) - 1);
    }
    
    // Update normalized cache
    float* cached = db->normalized_cache + db->num_entries * db->dimension;
    memcpy(cached, vector, db->dimension * sizeof(float));
    normalize_vector(cached, db->dimension);
    
    return db->num_entries++;
}

EXPORT int search_vectors(
    const VectorDB* db,
    const float* query,
    int top_k,
    double threshold,
    size_t* ids_out,
    double* scores_out
) {
    if (!db || db->num_entries == 0) return 0;
    
    // Normalize query
    std::vector<float> norm_query(query, query + db->dimension);
    normalize_vector(norm_query.data(), db->dimension);
    
    // Compute all similarities
    std::vector<std::pair<double, size_t>> results;
    
    #pragma omp parallel
    {
        std::vector<std::pair<double, size_t>> local_results;
        
        #pragma omp for nowait
        for (size_t i = 0; i < db->num_entries; i++) {
            float* cached = (float*)(db->normalized_cache + i * db->dimension);
            double sim = cosine_similarity_f(norm_query.data(), cached, db->dimension);
            
            if (sim >= threshold) {
                local_results.push_back({sim, i});
            }
        }
        
        #pragma omp critical
        {
            results.insert(results.end(), local_results.begin(), local_results.end());
        }
    }
    
    // Sort by similarity
    std::partial_sort(results.begin(), 
                      results.begin() + std::min((size_t)top_k, results.size()),
                      results.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    int count = std::min((size_t)top_k, results.size());
    for (int i = 0; i < count; i++) {
        ids_out[i] = results[i].second;
        scores_out[i] = results[i].first;
    }
    
    return count;
}

EXPORT int delete_vector(VectorDB* db, size_t id) {
    if (id >= db->num_entries) return 0;
    
    delete[] db->entries[id].vector;
    
    // Shift remaining entries
    for (size_t i = id; i < db->num_entries - 1; i++) {
        db->entries[i] = db->entries[i + 1];
        db->entries[i].id = i;
        memcpy(db->normalized_cache + i * db->dimension,
               db->normalized_cache + (i + 1) * db->dimension,
               db->dimension * sizeof(float));
    }
    
    db->num_entries--;
    return 1;
}

EXPORT int update_vector(VectorDB* db, size_t id, const float* new_vector) {
    if (id >= db->num_entries) return 0;
    
    memcpy(db->entries[id].vector, new_vector, db->dimension * sizeof(float));
    
    float* cached = db->normalized_cache + id * db->dimension;
    memcpy(cached, new_vector, db->dimension * sizeof(float));
    normalize_vector(cached, db->dimension);
    
    return 1;
}

// ============================================================================
// FLAT INDEX
// ============================================================================

EXPORT FlatIndex* create_flat_index(int dimension, size_t max_vectors) {
    FlatIndex* index = new FlatIndex();
    index->dimension = dimension;
    index->max_vectors = max_vectors;
    index->num_vectors = 0;
    index->vectors = new float[max_vectors * dimension];
    return index;
}

EXPORT void free_flat_index(FlatIndex* index) {
    if (index) {
        delete[] index->vectors;
        delete index;
    }
}

EXPORT void flat_index_add(FlatIndex* index, const float* vectors, size_t num_vectors) {
    size_t to_add = std::min(num_vectors, index->max_vectors - index->num_vectors);
    memcpy(index->vectors + index->num_vectors * index->dimension,
           vectors,
           to_add * index->dimension * sizeof(float));
    index->num_vectors += to_add;
}

EXPORT int flat_index_search(
    const FlatIndex* index,
    const float* query,
    int top_k,
    size_t* ids_out,
    float* distances_out
) {
    std::vector<std::pair<float, size_t>> results(index->num_vectors);
    
    #pragma omp parallel for
    for (size_t i = 0; i < index->num_vectors; i++) {
        const float* vec = index->vectors + i * index->dimension;
        results[i] = {l2_distance(query, vec, index->dimension), i};
    }
    
    std::partial_sort(results.begin(),
                      results.begin() + std::min((size_t)top_k, index->num_vectors),
                      results.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
    
    int count = std::min((size_t)top_k, index->num_vectors);
    for (int i = 0; i < count; i++) {
        ids_out[i] = results[i].second;
        distances_out[i] = results[i].first;
    }
    
    return count;
}

// ============================================================================
// IVF INDEX
// ============================================================================

EXPORT IVFIndex* create_ivf_index(
    int dimension,
    int num_centroids,
    const float* training_vectors,
    size_t num_training
) {
    IVFIndex* index = new IVFIndex();
    index->dimension = dimension;
    index->num_centroids = num_centroids;
    
    index->centroids = new float*[num_centroids];
    index->inverted_lists = new int*[num_centroids];
    index->list_sizes = new int[num_centroids];
    index->list_vectors = new float*[num_centroids];
    
    // Simple k-means initialization (use first k vectors as centroids)
    for (int c = 0; c < num_centroids; c++) {
        index->centroids[c] = new float[dimension];
        if (c < (int)num_training) {
            memcpy(index->centroids[c], training_vectors + c * dimension, dimension * sizeof(float));
        }
        index->list_sizes[c] = 0;
        index->inverted_lists[c] = nullptr;
        index->list_vectors[c] = nullptr;
    }
    
    return index;
}

EXPORT void free_ivf_index(IVFIndex* index) {
    if (index) {
        for (int c = 0; c < index->num_centroids; c++) {
            delete[] index->centroids[c];
            delete[] index->inverted_lists[c];
            delete[] index->list_vectors[c];
        }
        delete[] index->centroids;
        delete[] index->inverted_lists;
        delete[] index->list_sizes;
        delete[] index->list_vectors;
        delete index;
    }
}

EXPORT int ivf_index_search(
    const IVFIndex* index,
    const float* query,
    int top_k,
    int num_probes,
    size_t* ids_out,
    float* distances_out
) {
    // Find closest centroids
    std::vector<std::pair<float, int>> centroid_dists(index->num_centroids);
    for (int c = 0; c < index->num_centroids; c++) {
        centroid_dists[c] = {l2_distance(query, index->centroids[c], index->dimension), c};
    }
    
    std::partial_sort(centroid_dists.begin(),
                      centroid_dists.begin() + std::min(num_probes, index->num_centroids),
                      centroid_dists.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // Placeholder - would search inverted lists
    return 0;
}

// ============================================================================
// MEMORY COMPRESSION
// ============================================================================

EXPORT CompressedMemory* compress_memory(
    const float* data,
    size_t num_vectors,
    int dimension,
    int num_subquantizers,
    int bits_per_code
) {
    CompressedMemory* mem = new CompressedMemory();
    
    int codebook_size = 1 << bits_per_code;
    mem->codebook_size = codebook_size;
    mem->original_size = num_vectors * dimension * sizeof(float);
    
    // Create simple codebook
    mem->codebook = new float[codebook_size * dimension];
    for (int c = 0; c < codebook_size; c++) {
        for (int d = 0; d < dimension; d++) {
            mem->codebook[c * dimension + d] = (float)c / codebook_size;
        }
    }
    
    // Compress data (quantize to nearest codebook entry)
    mem->compressed_size = num_vectors * num_subquantizers;
    mem->compressed_data = new float[mem->compressed_size];
    
    for (size_t v = 0; v < num_vectors; v++) {
        // Simple quantization
        for (int s = 0; s < num_subquantizers; s++) {
            float val = data[v * dimension + s % dimension];
            int code = (int)(val * codebook_size) % codebook_size;
            mem->compressed_data[v * num_subquantizers + s] = (float)code;
        }
    }
    
    mem->compression_ratio = (double)mem->original_size / (mem->compressed_size * sizeof(float));
    
    return mem;
}

EXPORT void free_compressed_memory(CompressedMemory* mem) {
    if (mem) {
        delete[] mem->compressed_data;
        delete[] mem->codebook;
        delete mem;
    }
}

EXPORT void decompress_memory(
    const CompressedMemory* mem,
    float* output,
    size_t output_size
) {
    // Placeholder decompression
    memset(output, 0, output_size * sizeof(float));
}

EXPORT int search_compressed(
    const CompressedMemory* mem,
    const float* query,
    int dimension,
    int top_k,
    size_t* ids_out,
    float* distances_out
) {
    // Placeholder
    return 0;
}

// ============================================================================
// KD-TREE
// ============================================================================

static int build_kdtree_recursive(
    KDTree* tree,
    std::vector<int>& indices,
    int start,
    int end,
    int depth
) {
    if (start >= end) return -1;
    
    int node_idx = tree->num_nodes++;
    KDTreeNode* node = &tree->nodes[node_idx];
    
    int dim = depth % tree->dimension;
    node->split_dim = dim;
    
    // Sort by split dimension
    int mid = (start + end) / 2;
    std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end,
        [&](int a, int b) {
            return tree->points[a * tree->dimension + dim] < tree->points[b * tree->dimension + dim];
        });
    
    node->point_index = indices[mid];
    node->split_value = tree->points[indices[mid] * tree->dimension + dim];
    
    node->left_child = build_kdtree_recursive(tree, indices, start, mid, depth + 1);
    node->right_child = build_kdtree_recursive(tree, indices, mid + 1, end, depth + 1);
    
    return node_idx;
}

EXPORT KDTree* build_kdtree(const float* points, int num_points, int dimension) {
    KDTree* tree = new KDTree();
    tree->dimension = dimension;
    tree->num_points = num_points;
    tree->max_nodes = num_points * 2;
    tree->num_nodes = 0;
    
    tree->nodes = new KDTreeNode[tree->max_nodes];
    tree->points = new float[num_points * dimension];
    memcpy(tree->points, points, num_points * dimension * sizeof(float));
    
    std::vector<int> indices(num_points);
    for (int i = 0; i < num_points; i++) indices[i] = i;
    
    build_kdtree_recursive(tree, indices, 0, num_points, 0);
    
    return tree;
}

EXPORT void free_kdtree(KDTree* tree) {
    if (tree) {
        delete[] tree->nodes;
        delete[] tree->points;
        delete tree;
    }
}

static void kdtree_knn_recursive(
    const KDTree* tree,
    int node_idx,
    const float* query,
    std::priority_queue<std::pair<float, int>>& heap,
    int k
) {
    if (node_idx < 0) return;
    
    const KDTreeNode* node = &tree->nodes[node_idx];
    const float* point = tree->points + node->point_index * tree->dimension;
    
    float dist = l2_distance(query, point, tree->dimension);
    
    if ((int)heap.size() < k) {
        heap.push({dist, node->point_index});
    } else if (dist < heap.top().first) {
        heap.pop();
        heap.push({dist, node->point_index});
    }
    
    float diff = query[node->split_dim] - node->split_value;
    int first = (diff < 0) ? node->left_child : node->right_child;
    int second = (diff < 0) ? node->right_child : node->left_child;
    
    kdtree_knn_recursive(tree, first, query, heap, k);
    
    if ((int)heap.size() < k || std::abs(diff) < heap.top().first) {
        kdtree_knn_recursive(tree, second, query, heap, k);
    }
}

EXPORT int kdtree_knn(
    const KDTree* tree,
    const float* query,
    int k,
    int* indices_out,
    float* distances_out
) {
    std::priority_queue<std::pair<float, int>> heap;
    kdtree_knn_recursive(tree, 0, query, heap, k);
    
    int count = heap.size();
    for (int i = count - 1; i >= 0; i--) {
        indices_out[i] = heap.top().second;
        distances_out[i] = heap.top().first;
        heap.pop();
    }
    
    return count;
}

EXPORT int kdtree_range_search(
    const KDTree* tree,
    const float* query,
    float radius,
    int* indices_out,
    int max_results
) {
    // Simple range search using brute force
    int count = 0;
    for (int i = 0; i < tree->num_points && count < max_results; i++) {
        float dist = l2_distance(query, tree->points + i * tree->dimension, tree->dimension);
        if (dist <= radius) {
            indices_out[count++] = i;
        }
    }
    return count;
}

// ============================================================================
// LSH INDEX
// ============================================================================

EXPORT LSHIndex* create_lsh_index(int dimension, int num_tables, int num_hashes_per_table) {
    LSHIndex* index = new LSHIndex();
    index->dimension = dimension;
    index->num_tables = num_tables;
    index->num_hashes_per_table = num_hashes_per_table;
    
    index->hash_functions = new float*[num_tables];
    index->hash_tables = new int*[num_tables];
    index->table_sizes = new int[num_tables];
    
    std::mt19937 gen(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    
    for (int t = 0; t < num_tables; t++) {
        index->hash_functions[t] = new float[num_hashes_per_table * dimension];
        for (int h = 0; h < num_hashes_per_table * dimension; h++) {
            index->hash_functions[t][h] = normal(gen);
        }
        index->hash_tables[t] = nullptr;
        index->table_sizes[t] = 0;
    }
    
    return index;
}

EXPORT void free_lsh_index(LSHIndex* index) {
    if (index) {
        for (int t = 0; t < index->num_tables; t++) {
            delete[] index->hash_functions[t];
            delete[] index->hash_tables[t];
        }
        delete[] index->hash_functions;
        delete[] index->hash_tables;
        delete[] index->table_sizes;
        delete index;
    }
}

EXPORT void lsh_index_add(LSHIndex* index, const float* vector, int id) {
    // Placeholder - would compute hash and add to tables
}

EXPORT int lsh_index_query(
    const LSHIndex* index,
    const float* query,
    int* candidates_out,
    int max_candidates
) {
    // Placeholder
    return 0;
}

// ============================================================================
// EPISODIC MEMORY
// ============================================================================

EXPORT EpisodicMemory* create_episodic_memory(int max_episodes, int dimension) {
    EpisodicMemory* mem = new EpisodicMemory();
    mem->max_episodes = max_episodes;
    mem->num_episodes = 0;
    mem->dimension = dimension;
    
    mem->episodes = new Episode[max_episodes];
    mem->context_index = new float[max_episodes * dimension];
    
    for (int i = 0; i < max_episodes; i++) {
        mem->episodes[i].context_vector = nullptr;
        mem->episodes[i].state_vector = nullptr;
        mem->episodes[i].action_vector = nullptr;
    }
    
    return mem;
}

EXPORT void free_episodic_memory(EpisodicMemory* mem) {
    if (mem) {
        for (int i = 0; i < mem->num_episodes; i++) {
            delete[] mem->episodes[i].context_vector;
            delete[] mem->episodes[i].state_vector;
            delete[] mem->episodes[i].action_vector;
        }
        delete[] mem->episodes;
        delete[] mem->context_index;
        delete mem;
    }
}

EXPORT int add_episode(
    EpisodicMemory* mem,
    const float* context,
    const float* state,
    const float* action,
    float reward
) {
    if (mem->num_episodes >= mem->max_episodes) return -1;
    
    Episode* ep = &mem->episodes[mem->num_episodes];
    ep->id = mem->num_episodes;
    ep->reward = reward;
    ep->timestamp = 0;
    ep->next_episode_id = -1;
    
    ep->context_vector = new float[mem->dimension];
    ep->state_vector = new float[mem->dimension];
    ep->action_vector = new float[mem->dimension];
    
    memcpy(ep->context_vector, context, mem->dimension * sizeof(float));
    memcpy(ep->state_vector, state, mem->dimension * sizeof(float));
    memcpy(ep->action_vector, action, mem->dimension * sizeof(float));
    
    memcpy(mem->context_index + mem->num_episodes * mem->dimension,
           context, mem->dimension * sizeof(float));
    
    return mem->num_episodes++;
}

EXPORT int recall_episode(
    const EpisodicMemory* mem,
    const float* context,
    float* state_out,
    float* action_out
) {
    float best_sim = -1.0f;
    int best_idx = -1;
    
    for (int i = 0; i < mem->num_episodes; i++) {
        float sim = cosine_similarity_f(context, mem->episodes[i].context_vector, mem->dimension);
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = i;
        }
    }
    
    if (best_idx >= 0) {
        memcpy(state_out, mem->episodes[best_idx].state_vector, mem->dimension * sizeof(float));
        memcpy(action_out, mem->episodes[best_idx].action_vector, mem->dimension * sizeof(float));
        return 1;
    }
    
    return 0;
}

EXPORT void consolidate_episodic_memory(EpisodicMemory* mem, float similarity_threshold) {
    std::vector<bool> keep(mem->num_episodes, true);
    
    for (int i = 0; i < mem->num_episodes; i++) {
        if (!keep[i]) continue;
        for (int j = i + 1; j < mem->num_episodes; j++) {
            if (!keep[j]) continue;
            float sim = cosine_similarity_f(mem->episodes[i].context_vector,
                                            mem->episodes[j].context_vector,
                                            mem->dimension);
            if (sim > similarity_threshold) {
                keep[j] = false;
            }
        }
    }
    
    // Would compact the array here
}

EXPORT const char* get_memory_version() {
    return MEMORY_VERSION;
}
