
#ifndef MEMORY_H
#define MEMORY_H

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

#include <cstdint>

extern "C" {
    // ============================================================================
    // MEMORY CORE - AGI C++ Implementation
    // Ultra-fast vector database, similarity search, memory compression
    // ============================================================================

    // ============================================================================
    // VECTOR DATABASE
    // ============================================================================
    
    struct VectorEntry {
        size_t id;
        float* vector;
        int dimension;
        int64_t timestamp;
        double importance;
        char metadata[128];
    };
    
    struct VectorDB {
        VectorEntry* entries;
        size_t num_entries;
        size_t max_entries;
        int dimension;
        float* normalized_cache;
    };
    
    // Create vector database
    EXPORT VectorDB* create_vector_db(int dimension, size_t max_vectors);
    
    // Free vector database
    EXPORT void free_vector_db(VectorDB* db);
    
    // Add vector to database
    EXPORT size_t add_vector(
        VectorDB* db,
        const float* vector,
        const char* metadata
    );
    
    // Search similar vectors
    EXPORT int search_vectors(
        const VectorDB* db,
        const float* query,
        int top_k,
        double threshold,
        size_t* ids_out,
        double* scores_out
    );
    
    // Delete vector by ID
    EXPORT int delete_vector(VectorDB* db, size_t id);
    
    // Update vector
    EXPORT int update_vector(
        VectorDB* db,
        size_t id,
        const float* new_vector
    );

    // ============================================================================
    // FAISS-LIKE INDEX (Simplified)
    // ============================================================================
    
    struct FlatIndex {
        float* vectors;
        size_t num_vectors;
        size_t max_vectors;
        int dimension;
    };
    
    struct IVFIndex {
        float** centroids;
        int num_centroids;
        int** inverted_lists;
        int* list_sizes;
        float** list_vectors;
        int dimension;
    };
    
    // Create flat index (brute force)
    EXPORT FlatIndex* create_flat_index(int dimension, size_t max_vectors);
    
    // Free flat index
    EXPORT void free_flat_index(FlatIndex* index);
    
    // Add vectors to flat index
    EXPORT void flat_index_add(FlatIndex* index, const float* vectors, size_t num_vectors);
    
    // Search flat index
    EXPORT int flat_index_search(
        const FlatIndex* index,
        const float* query,
        int top_k,
        size_t* ids_out,
        float* distances_out
    );
    
    // Create IVF index
    EXPORT IVFIndex* create_ivf_index(
        int dimension,
        int num_centroids,
        const float* training_vectors,
        size_t num_training
    );
    
    // Free IVF index
    EXPORT void free_ivf_index(IVFIndex* index);
    
    // IVF search
    EXPORT int ivf_index_search(
        const IVFIndex* index,
        const float* query,
        int top_k,
        int num_probes,
        size_t* ids_out,
        float* distances_out
    );

    // ============================================================================
    // MEMORY COMPRESSION
    // ============================================================================
    
    struct CompressedMemory {
        float* compressed_data;
        size_t compressed_size;
        size_t original_size;
        float* codebook;
        int codebook_size;
        double compression_ratio;
    };
    
    // Compress memory (PQ-like)
    EXPORT CompressedMemory* compress_memory(
        const float* data,
        size_t num_vectors,
        int dimension,
        int num_subquantizers,
        int bits_per_code
    );
    
    // Free compressed memory
    EXPORT void free_compressed_memory(CompressedMemory* mem);
    
    // Decompress memory
    EXPORT void decompress_memory(
        const CompressedMemory* mem,
        float* output,
        size_t output_size
    );
    
    // Search compressed memory
    EXPORT int search_compressed(
        const CompressedMemory* mem,
        const float* query,
        int dimension,
        int top_k,
        size_t* ids_out,
        float* distances_out
    );

    // ============================================================================
    // MEMORY INDEXING
    // ============================================================================
    
    struct KDTreeNode {
        int split_dim;
        float split_value;
        int left_child;
        int right_child;
        int point_index;
    };
    
    struct KDTree {
        KDTreeNode* nodes;
        int num_nodes;
        int max_nodes;
        float* points;
        int num_points;
        int dimension;
    };
    
    // Build KD-tree
    EXPORT KDTree* build_kdtree(
        const float* points,
        int num_points,
        int dimension
    );
    
    // Free KD-tree
    EXPORT void free_kdtree(KDTree* tree);
    
    // KNN search in KD-tree
    EXPORT int kdtree_knn(
        const KDTree* tree,
        const float* query,
        int k,
        int* indices_out,
        float* distances_out
    );
    
    // Range search in KD-tree
    EXPORT int kdtree_range_search(
        const KDTree* tree,
        const float* query,
        float radius,
        int* indices_out,
        int max_results
    );

    // ============================================================================
    // LSH (Locality Sensitive Hashing)
    // ============================================================================
    
    struct LSHIndex {
        float** hash_functions;
        int num_tables;
        int num_hashes_per_table;
        int dimension;
        int** hash_tables;
        int* table_sizes;
    };
    
    // Create LSH index
    EXPORT LSHIndex* create_lsh_index(
        int dimension,
        int num_tables,
        int num_hashes_per_table
    );
    
    // Free LSH index
    EXPORT void free_lsh_index(LSHIndex* index);
    
    // Add to LSH index
    EXPORT void lsh_index_add(
        LSHIndex* index,
        const float* vector,
        int id
    );
    
    // Query LSH index
    EXPORT int lsh_index_query(
        const LSHIndex* index,
        const float* query,
        int* candidates_out,
        int max_candidates
    );

    // ============================================================================
    // EPISODIC MEMORY
    // ============================================================================
    
    struct Episode {
        int id;
        float* context_vector;
        float* state_vector;
        float* action_vector;
        float reward;
        int64_t timestamp;
        int next_episode_id;
    };
    
    struct EpisodicMemory {
        Episode* episodes;
        int num_episodes;
        int max_episodes;
        int dimension;
        float* context_index;
    };
    
    // Create episodic memory
    EXPORT EpisodicMemory* create_episodic_memory(int max_episodes, int dimension);
    
    // Free episodic memory
    EXPORT void free_episodic_memory(EpisodicMemory* mem);
    
    // Add episode
    EXPORT int add_episode(
        EpisodicMemory* mem,
        const float* context,
        const float* state,
        const float* action,
        float reward
    );
    
    // Recall similar episode
    EXPORT int recall_episode(
        const EpisodicMemory* mem,
        const float* context,
        float* state_out,
        float* action_out
    );
    
    // Consolidate memories (remove similar)
    EXPORT void consolidate_episodic_memory(
        EpisodicMemory* mem,
        float similarity_threshold
    );

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================
    
    // Compute L2 distance
    EXPORT float l2_distance(const float* a, const float* b, int dimension);
    
    // Compute cosine similarity
    EXPORT float cosine_similarity_f(const float* a, const float* b, int dimension);
    
    // Compute dot product
    EXPORT float dot_product(const float* a, const float* b, int dimension);
    
    // Normalize vector
    EXPORT void normalize_vector(float* vector, int dimension);
    
    EXPORT const char* get_memory_version();
}

#endif
