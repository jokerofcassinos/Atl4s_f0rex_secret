
#include "reasoning.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <random>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// GLOBAL STATE
// ============================================================================

static const char* REASONING_VERSION = "1.0.0-AGI";
static thread_local std::mt19937 tl_gen(std::random_device{}());
static double g_total_reasoning_time_ms = 0.0;
static int g_reasoning_calls = 0;

// ============================================================================
// THOUGHT TREE IMPLEMENTATION
// ============================================================================

EXPORT ThoughtTree* create_thought_tree(int max_nodes) {
    ThoughtTree* tree = new ThoughtTree();
    tree->max_nodes = max_nodes;
    tree->num_nodes = 0;
    tree->root_id = -1;
    tree->nodes = new ThoughtNode[max_nodes];
    
    for (int i = 0; i < max_nodes; i++) {
        tree->nodes[i].id = -1;
        tree->nodes[i].child_ids = nullptr;
    }
    
    return tree;
}

EXPORT void free_thought_tree(ThoughtTree* tree) {
    if (tree) {
        for (int i = 0; i < tree->num_nodes; i++) {
            if (tree->nodes[i].child_ids) {
                delete[] tree->nodes[i].child_ids;
            }
        }
        delete[] tree->nodes;
        delete tree;
    }
}

EXPORT int add_thought_node(
    ThoughtTree* tree,
    int parent_id,
    const char* question,
    const char* answer,
    double confidence
) {
    if (tree->num_nodes >= tree->max_nodes) return -1;
    
    int id = tree->num_nodes;
    ThoughtNode* node = &tree->nodes[id];
    
    node->id = id;
    node->parent_id = parent_id;
    strncpy(node->question, question, sizeof(node->question) - 1);
    strncpy(node->answer, answer, sizeof(node->answer) - 1);
    node->confidence = confidence;
    node->relevance = 1.0;
    node->num_children = 0;
    node->child_ids = nullptr;
    
    // Calculate depth
    if (parent_id < 0) {
        node->depth = 0;
        tree->root_id = id;
    } else {
        node->depth = tree->nodes[parent_id].depth + 1;
        
        // Add to parent's children
        ThoughtNode* parent = &tree->nodes[parent_id];
        int* new_children = new int[parent->num_children + 1];
        if (parent->child_ids) {
            memcpy(new_children, parent->child_ids, parent->num_children * sizeof(int));
            delete[] parent->child_ids;
        }
        new_children[parent->num_children] = id;
        parent->child_ids = new_children;
        parent->num_children++;
    }
    
    tree->num_nodes++;
    return id;
}

EXPORT ThoughtNode* get_thought_node(ThoughtTree* tree, int node_id) {
    if (!tree || node_id < 0 || node_id >= tree->num_nodes) return nullptr;
    return &tree->nodes[node_id];
}

EXPORT int get_thought_children(ThoughtTree* tree, int node_id, int* children_out, int max_children) {
    ThoughtNode* node = get_thought_node(tree, node_id);
    if (!node) return 0;
    
    int count = std::min(node->num_children, max_children);
    if (node->child_ids) {
        memcpy(children_out, node->child_ids, count * sizeof(int));
    }
    return count;
}

EXPORT int traverse_thought_tree_dfs(ThoughtTree* tree, int* path_out, int max_path) {
    if (!tree || tree->root_id < 0) return 0;
    
    std::vector<int> path;
    std::vector<int> stack;
    stack.push_back(tree->root_id);
    
    while (!stack.empty() && path.size() < (size_t)max_path) {
        int current = stack.back();
        stack.pop_back();
        path.push_back(current);
        
        ThoughtNode* node = &tree->nodes[current];
        for (int i = node->num_children - 1; i >= 0; i--) {
            stack.push_back(node->child_ids[i]);
        }
    }
    
    int count = std::min((int)path.size(), max_path);
    memcpy(path_out, path.data(), count * sizeof(int));
    return count;
}

EXPORT double find_best_thought_path(ThoughtTree* tree, int* path_out, int max_path, int* path_length) {
    if (!tree || tree->root_id < 0) {
        *path_length = 0;
        return 0.0;
    }
    
    std::vector<int> best_path;
    double best_confidence = -1.0;
    
    // BFS to find path with highest cumulative confidence
    std::queue<std::pair<int, std::vector<int>>> queue;
    queue.push({tree->root_id, {tree->root_id}});
    
    while (!queue.empty()) {
        auto [node_id, current_path] = queue.front();
        queue.pop();
        
        ThoughtNode* node = &tree->nodes[node_id];
        
        if (node->num_children == 0) {
            // Leaf node - calculate path confidence
            double conf = 0.0;
            for (int id : current_path) {
                conf += tree->nodes[id].confidence;
            }
            conf /= current_path.size();
            
            if (conf > best_confidence) {
                best_confidence = conf;
                best_path = current_path;
            }
        } else {
            for (int i = 0; i < node->num_children; i++) {
                std::vector<int> new_path = current_path;
                new_path.push_back(node->child_ids[i]);
                queue.push({node->child_ids[i], new_path});
            }
        }
    }
    
    *path_length = std::min((int)best_path.size(), max_path);
    memcpy(path_out, best_path.data(), *path_length * sizeof(int));
    return best_confidence;
}

// ============================================================================
// INFINITE WHY ENGINE
// ============================================================================

EXPORT ReasoningResult* deep_scan_recursive(
    const ReasoningEvent* root_event,
    int max_depth,
    int max_branches,
    double confidence_threshold
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    ReasoningResult* result = new ReasoningResult();
    result->tree = create_thought_tree(max_depth * max_branches + 1);
    result->chain = new ReasoningChain();
    result->chain->events = new ReasoningEvent[max_depth];
    result->chain->num_events = 0;
    result->chain->max_depth = max_depth;
    
    // Add root event
    int root_id = add_thought_node(result->tree, -1, "Root Event", root_event->description, 1.0);
    result->chain->events[0] = *root_event;
    result->chain->num_events++;
    
    // Recursive expansion (simulated)
    std::queue<std::pair<int, int>> queue;  // (node_id, depth)
    queue.push({root_id, 0});
    int iterations = 0;
    
    while (!queue.empty() && iterations < max_depth * max_branches) {
        auto [node_id, depth] = queue.front();
        queue.pop();
        
        if (depth >= max_depth) continue;
        
        // Generate child questions (simulated)
        int num_children = std::min(max_branches, 3);
        for (int i = 0; i < num_children; i++) {
            char question[256];
            char answer[512];
            snprintf(question, sizeof(question), "Why did event %d occur at level %d?", node_id, depth);
            snprintf(answer, sizeof(answer), "Cause %d at depth %d with confidence %.2f", i, depth, 0.8 - depth * 0.1);
            
            double conf = 0.8 - depth * 0.1 - i * 0.05;
            if (conf < confidence_threshold) continue;
            
            int child_id = add_thought_node(result->tree, node_id, question, answer, conf);
            if (child_id >= 0) {
                queue.push({child_id, depth + 1});
            }
        }
        iterations++;
    }
    
    result->iterations = iterations;
    result->final_confidence = 0.8;
    
    auto end = std::chrono::high_resolution_clock::now();
    result->time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    g_total_reasoning_time_ms += result->time_ms;
    g_reasoning_calls++;
    
    return result;
}

EXPORT void free_reasoning_result(ReasoningResult* result) {
    if (result) {
        free_thought_tree(result->tree);
        if (result->chain) {
            delete[] result->chain->events;
            delete result->chain;
        }
        delete result;
    }
}

EXPORT double reason_step(
    const ReasoningEvent* event,
    const char* question,
    char* answer_out,
    int answer_max_len
) {
    // Simple reasoning simulation
    snprintf(answer_out, answer_max_len, 
             "Analysis of event '%s' regarding '%s': Magnitude %.2f suggests moderate impact.",
             event->description, question, event->magnitude);
    return 0.75;
}

// ============================================================================
// PATTERN MATCHING
// ============================================================================

EXPORT int find_similar_patterns(
    const double* query_vector,
    int query_size,
    const double* pattern_database,
    int num_patterns,
    int pattern_size,
    double threshold,
    PatternMatch* matches_out,
    int max_matches
) {
    int match_count = 0;
    
    #pragma omp parallel for
    for (int p = 0; p < num_patterns; p++) {
        if (match_count >= max_matches) continue;
        
        const double* pattern = pattern_database + p * pattern_size;
        
        // Compute cosine similarity
        double dot = 0.0, norm_q = 0.0, norm_p = 0.0;
        int min_size = std::min(query_size, pattern_size);
        
        for (int i = 0; i < min_size; i++) {
            dot += query_vector[i] * pattern[i];
            norm_q += query_vector[i] * query_vector[i];
            norm_p += pattern[i] * pattern[i];
        }
        
        double similarity = dot / (std::sqrt(norm_q) * std::sqrt(norm_p) + 1e-10);
        
        if (similarity >= threshold) {
            #pragma omp critical
            {
                if (match_count < max_matches) {
                    matches_out[match_count].pattern_id = p;
                    matches_out[match_count].start_index = 0;
                    matches_out[match_count].length = pattern_size;
                    matches_out[match_count].similarity = similarity;
                    matches_out[match_count].confidence = similarity;
                    match_count++;
                }
            }
        }
    }
    
    return match_count;
}

EXPORT double dtw_similarity(
    const double* seq1,
    int len1,
    const double* seq2,
    int len2
) {
    // Dynamic Time Warping
    std::vector<std::vector<double>> dtw(len1 + 1, std::vector<double>(len2 + 1, 1e9));
    dtw[0][0] = 0.0;
    
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            double cost = std::abs(seq1[i-1] - seq2[j-1]);
            dtw[i][j] = cost + std::min({dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1]});
        }
    }
    
    double distance = dtw[len1][len2];
    double max_dist = std::max(len1, len2) * 100.0;  // Normalize
    return 1.0 - (distance / max_dist);
}

EXPORT int simd_pattern_search(
    const float* data,
    int data_length,
    const float* pattern,
    int pattern_length,
    float threshold,
    int* matches_out,
    int max_matches
) {
    int match_count = 0;
    
    #pragma omp parallel for
    for (int i = 0; i <= data_length - pattern_length; i++) {
        float dot = 0.0f, norm_d = 0.0f, norm_p = 0.0f;
        
        // SIMD-friendly loop
        for (int j = 0; j < pattern_length; j++) {
            float d = data[i + j];
            float p = pattern[j];
            dot += d * p;
            norm_d += d * d;
            norm_p += p * p;
        }
        
        float sim = dot / (std::sqrt(norm_d) * std::sqrt(norm_p) + 1e-10f);
        
        if (sim >= threshold) {
            #pragma omp critical
            {
                if (match_count < max_matches) {
                    matches_out[match_count++] = i;
                }
            }
        }
    }
    
    return match_count;
}

// ============================================================================
// CAUSAL CHAIN
// ============================================================================

EXPORT CausalGraph* create_causal_graph(int max_links) {
    CausalGraph* graph = new CausalGraph();
    graph->max_links = max_links;
    graph->num_links = 0;
    graph->num_nodes = 0;
    graph->links = new CausalLink[max_links];
    return graph;
}

EXPORT void free_causal_graph(CausalGraph* graph) {
    if (graph) {
        delete[] graph->links;
        delete graph;
    }
}

EXPORT int add_causal_link(
    CausalGraph* graph,
    int cause_id,
    int effect_id,
    double strength,
    double delay
) {
    if (graph->num_links >= graph->max_links) return -1;
    
    CausalLink* link = &graph->links[graph->num_links];
    link->cause_id = cause_id;
    link->effect_id = effect_id;
    link->strength = strength;
    link->delay = delay;
    link->link_type = 0;
    
    graph->num_nodes = std::max(graph->num_nodes, std::max(cause_id, effect_id) + 1);
    
    return graph->num_links++;
}

EXPORT int traverse_causal_chain(
    const CausalGraph* graph,
    int start_node,
    int* path_out,
    int max_path,
    double min_strength
) {
    std::vector<int> path;
    std::vector<bool> visited(graph->num_nodes, false);
    
    std::queue<int> queue;
    queue.push(start_node);
    visited[start_node] = true;
    
    while (!queue.empty() && path.size() < (size_t)max_path) {
        int current = queue.front();
        queue.pop();
        path.push_back(current);
        
        for (int i = 0; i < graph->num_links; i++) {
            CausalLink* link = &graph->links[i];
            if (link->cause_id == current && link->strength >= min_strength) {
                if (!visited[link->effect_id]) {
                    visited[link->effect_id] = true;
                    queue.push(link->effect_id);
                }
            }
        }
    }
    
    int count = std::min((int)path.size(), max_path);
    memcpy(path_out, path.data(), count * sizeof(int));
    return count;
}

EXPORT double calculate_causal_effect(
    const CausalGraph* graph,
    int cause_id,
    int effect_id
) {
    // Simple path-based causal effect calculation
    std::vector<double> effects(graph->num_nodes, 0.0);
    effects[cause_id] = 1.0;
    
    // Forward propagation
    for (int iter = 0; iter < graph->num_nodes; iter++) {
        for (int i = 0; i < graph->num_links; i++) {
            CausalLink* link = &graph->links[i];
            effects[link->effect_id] = std::max(effects[link->effect_id],
                                                  effects[link->cause_id] * link->strength);
        }
    }
    
    return effects[effect_id];
}

EXPORT int find_root_causes(
    const CausalGraph* graph,
    int effect_id,
    int* causes_out,
    int max_causes
) {
    std::vector<int> root_causes;
    std::vector<bool> has_cause(graph->num_nodes, false);
    
    // Find nodes that have no incoming links
    for (int i = 0; i < graph->num_links; i++) {
        has_cause[graph->links[i].effect_id] = true;
    }
    
    // Trace back from effect
    std::vector<bool> visited(graph->num_nodes, false);
    std::queue<int> queue;
    queue.push(effect_id);
    
    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();
        
        if (visited[current]) continue;
        visited[current] = true;
        
        bool is_root = true;
        for (int i = 0; i < graph->num_links; i++) {
            if (graph->links[i].effect_id == current) {
                is_root = false;
                queue.push(graph->links[i].cause_id);
            }
        }
        
        if (is_root && current != effect_id) {
            root_causes.push_back(current);
        }
    }
    
    int count = std::min((int)root_causes.size(), max_causes);
    memcpy(causes_out, root_causes.data(), count * sizeof(int));
    return count;
}

// ============================================================================
// INFERENCE ENGINE
// ============================================================================

EXPORT InferenceEngine* create_inference_engine(int max_rules) {
    InferenceEngine* engine = new InferenceEngine();
    engine->max_rules = max_rules;
    engine->num_rules = 0;
    engine->rules = new Rule[max_rules];
    return engine;
}

EXPORT void free_inference_engine(InferenceEngine* engine) {
    if (engine) {
        delete[] engine->rules;
        delete engine;
    }
}

EXPORT int add_inference_rule(
    InferenceEngine* engine,
    const char* condition,
    const char* action,
    double confidence,
    int priority
) {
    if (engine->num_rules >= engine->max_rules) return -1;
    
    Rule* rule = &engine->rules[engine->num_rules];
    rule->id = engine->num_rules;
    strncpy(rule->condition, condition, sizeof(rule->condition) - 1);
    strncpy(rule->action, action, sizeof(rule->action) - 1);
    rule->confidence = confidence;
    rule->priority = priority;
    
    return engine->num_rules++;
}

EXPORT int forward_chain(
    InferenceEngine* engine,
    const char* facts,
    char* conclusions_out,
    int max_conclusions_len
) {
    // Simple forward chaining (substring matching)
    std::string conclusions;
    
    for (int i = 0; i < engine->num_rules; i++) {
        Rule* rule = &engine->rules[i];
        if (strstr(facts, rule->condition) != nullptr) {
            conclusions += rule->action;
            conclusions += "; ";
        }
    }
    
    strncpy(conclusions_out, conclusions.c_str(), max_conclusions_len - 1);
    return conclusions.length();
}

EXPORT int backward_chain(
    InferenceEngine* engine,
    const char* goal,
    char* required_facts_out,
    int max_facts_len
) {
    // Simple backward chaining
    std::string required;
    
    for (int i = 0; i < engine->num_rules; i++) {
        Rule* rule = &engine->rules[i];
        if (strstr(rule->action, goal) != nullptr) {
            required += rule->condition;
            required += "; ";
        }
    }
    
    strncpy(required_facts_out, required.c_str(), max_facts_len - 1);
    return required.length();
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

EXPORT const char* get_reasoning_version() {
    return REASONING_VERSION;
}

EXPORT void reset_reasoning_stats() {
    g_total_reasoning_time_ms = 0.0;
    g_reasoning_calls = 0;
}

EXPORT double get_reasoning_avg_time_ms() {
    return (g_reasoning_calls > 0) ? g_total_reasoning_time_ms / g_reasoning_calls : 0.0;
}
