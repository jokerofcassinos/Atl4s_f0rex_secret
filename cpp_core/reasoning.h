
#ifndef REASONING_H
#define REASONING_H

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

#include <cstdint>

extern "C" {
    // ============================================================================
    // REASONING CORE - AGI C++ Implementation
    // High-performance reasoning for InfiniteWhyEngine and ThoughtTree
    // ============================================================================

    // ============================================================================
    // THOUGHT TREE
    // ============================================================================
    
    struct ThoughtNode {
        int id;
        int parent_id;
        char question[256];
        char answer[512];
        double confidence;
        double relevance;
        int depth;
        int num_children;
        int* child_ids;
    };
    
    struct ThoughtTree {
        ThoughtNode* nodes;
        int num_nodes;
        int max_nodes;
        int root_id;
    };
    
    // Create thought tree
    EXPORT ThoughtTree* create_thought_tree(int max_nodes);
    
    // Free thought tree
    EXPORT void free_thought_tree(ThoughtTree* tree);
    
    // Add thought node
    EXPORT int add_thought_node(
        ThoughtTree* tree,
        int parent_id,
        const char* question,
        const char* answer,
        double confidence
    );
    
    // Get node by ID
    EXPORT ThoughtNode* get_thought_node(ThoughtTree* tree, int node_id);
    
    // Get children of node
    EXPORT int get_thought_children(ThoughtTree* tree, int node_id, int* children_out, int max_children);
    
    // Traverse tree depth-first
    EXPORT int traverse_thought_tree_dfs(ThoughtTree* tree, int* path_out, int max_path);
    
    // Find best path (highest confidence)
    EXPORT double find_best_thought_path(ThoughtTree* tree, int* path_out, int max_path, int* path_length);

    // ============================================================================
    // INFINITE WHY ENGINE (C++ Implementation)
    // ============================================================================
    
    struct ReasoningEvent {
        int id;
        char description[256];
        double magnitude;
        int64_t timestamp;
        int category;
        double* feature_vector;
        int feature_size;
    };
    
    struct ReasoningChain {
        ReasoningEvent* events;
        int num_events;
        double total_confidence;
        int max_depth;
    };
    
    struct ReasoningResult {
        ThoughtTree* tree;
        ReasoningChain* chain;
        double final_confidence;
        int iterations;
        double time_ms;
    };
    
    // Deep recursive scan
    EXPORT ReasoningResult* deep_scan_recursive(
        const ReasoningEvent* root_event,
        int max_depth,
        int max_branches,
        double confidence_threshold
    );
    
    // Free reasoning result
    EXPORT void free_reasoning_result(ReasoningResult* result);
    
    // Single step reasoning
    EXPORT double reason_step(
        const ReasoningEvent* event,
        const char* question,
        char* answer_out,
        int answer_max_len
    );

    // ============================================================================
    // PATTERN MATCHING (SIMD Optimized)
    // ============================================================================
    
    struct PatternMatch {
        int pattern_id;
        int start_index;
        int length;
        double similarity;
        double confidence;
    };
    
    // Find similar patterns in data
    EXPORT int find_similar_patterns(
        const double* query_vector,
        int query_size,
        const double* pattern_database,
        int num_patterns,
        int pattern_size,
        double threshold,
        PatternMatch* matches_out,
        int max_matches
    );
    
    // Match pattern using DTW (Dynamic Time Warping)
    EXPORT double dtw_similarity(
        const double* seq1,
        int len1,
        const double* seq2,
        int len2
    );
    
    // Fast pattern search with SIMD
    EXPORT int simd_pattern_search(
        const float* data,
        int data_length,
        const float* pattern,
        int pattern_length,
        float threshold,
        int* matches_out,
        int max_matches
    );

    // ============================================================================
    // CAUSAL CHAIN
    // ============================================================================
    
    struct CausalLink {
        int cause_id;
        int effect_id;
        double strength;
        double delay;
        int link_type;  // 0=direct, 1=indirect, 2=probabilistic
    };
    
    struct CausalGraph {
        CausalLink* links;
        int num_links;
        int max_links;
        int num_nodes;
    };
    
    // Create causal graph
    EXPORT CausalGraph* create_causal_graph(int max_links);
    
    // Free causal graph
    EXPORT void free_causal_graph(CausalGraph* graph);
    
    // Add causal link
    EXPORT int add_causal_link(
        CausalGraph* graph,
        int cause_id,
        int effect_id,
        double strength,
        double delay
    );
    
    // Traverse causal chain
    EXPORT int traverse_causal_chain(
        const CausalGraph* graph,
        int start_node,
        int* path_out,
        int max_path,
        double min_strength
    );
    
    // Calculate total causal effect
    EXPORT double calculate_causal_effect(
        const CausalGraph* graph,
        int cause_id,
        int effect_id
    );
    
    // Find root causes
    EXPORT int find_root_causes(
        const CausalGraph* graph,
        int effect_id,
        int* causes_out,
        int max_causes
    );

    // ============================================================================
    // INFERENCE ENGINE
    // ============================================================================
    
    struct Rule {
        int id;
        char condition[256];
        char action[256];
        double confidence;
        int priority;
    };
    
    struct InferenceEngine {
        Rule* rules;
        int num_rules;
        int max_rules;
    };
    
    // Create inference engine
    EXPORT InferenceEngine* create_inference_engine(int max_rules);
    
    // Free inference engine
    EXPORT void free_inference_engine(InferenceEngine* engine);
    
    // Add rule
    EXPORT int add_inference_rule(
        InferenceEngine* engine,
        const char* condition,
        const char* action,
        double confidence,
        int priority
    );
    
    // Run forward chaining
    EXPORT int forward_chain(
        InferenceEngine* engine,
        const char* facts,
        char* conclusions_out,
        int max_conclusions_len
    );
    
    // Run backward chaining
    EXPORT int backward_chain(
        InferenceEngine* engine,
        const char* goal,
        char* required_facts_out,
        int max_facts_len
    );

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================
    
    EXPORT const char* get_reasoning_version();
    EXPORT void reset_reasoning_stats();
    EXPORT double get_reasoning_avg_time_ms();
}

#endif
