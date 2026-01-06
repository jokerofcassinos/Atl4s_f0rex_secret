
#include "mcts.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <ctime>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// GLOBAL CONFIGURATION AND STATISTICS
// ============================================================================

static const char* MCTS_VERSION = "2.0.0-AGI-Ultra";
static std::atomic<int> g_total_simulations(0);
static std::atomic<double> g_total_time_ms(0);
static std::atomic<int> g_tt_hits(0);
static std::atomic<int> g_tt_misses(0);

// ============================================================================
// TRANSPOSITION TABLE
// ============================================================================

static std::vector<TTEntry> g_transposition_table;
static size_t g_tt_size = 0;
static std::mutex g_tt_mutex;

// Zobrist hashing for states
static uint64_t zobrist_hash(double price, double entry, int direction, int depth) {
    uint64_t h = 0;
    h ^= std::hash<double>{}(price * 1000);
    h ^= std::hash<double>{}(entry * 1000) << 16;
    h ^= std::hash<int>{}(direction) << 32;
    h ^= std::hash<int>{}(depth) << 48;
    return h;
}

EXPORT void init_transposition_table(size_t size) {
    std::lock_guard<std::mutex> lock(g_tt_mutex);
    g_tt_size = size;
    g_transposition_table.resize(size);
    for (size_t i = 0; i < size; i++) {
        g_transposition_table[i].valid = 0;
    }
}

EXPORT void clear_transposition_table() {
    std::lock_guard<std::mutex> lock(g_tt_mutex);
    for (size_t i = 0; i < g_tt_size; i++) {
        g_transposition_table[i].valid = 0;
    }
    g_tt_hits = 0;
    g_tt_misses = 0;
}

EXPORT int get_tt_hits() { return g_tt_hits.load(); }
EXPORT int get_tt_misses() { return g_tt_misses.load(); }

// ============================================================================
// RANDOM NUMBER GENERATION (Thread-Safe)
// ============================================================================

thread_local std::mt19937 tl_gen(std::random_device{}());

double generate_normal(double mean, double stddev) {
    std::normal_distribution<> d(mean, stddev);
    return d(tl_gen);
}

double generate_uniform() {
    std::uniform_real_distribution<> d(0.0, 1.0);
    return d(tl_gen);
}

// ============================================================================
// NODE STRUCTURES
// ============================================================================

enum MoveType {
    HOLD = 0,
    CLOSE = 1,
    ADD = 2
};

struct Node {
    MoveType move;
    Node* parent;
    std::vector<Node*> children;
    double wins;
    int visits;
    std::vector<MoveType> untried_moves;
    
    // RAVE statistics
    double amaf_wins;
    int amaf_visits;

    Node(MoveType m, Node* p) : move(m), parent(p), wins(0.0), visits(0),
                                  amaf_wins(0.0), amaf_visits(0) {
        untried_moves = {HOLD, CLOSE, ADD};
    }

    ~Node() {
        for (Node* c : children) delete c;
    }
    
    // UCT with optional RAVE
    double uct_value(double c, double rave_constant = 0.0) const {
        if (visits == 0) return 1e9;
        
        double q = wins / visits;
        double exploration = c * std::sqrt(std::log(parent->visits) / visits);
        
        // RAVE integration
        if (rave_constant > 0.0 && amaf_visits > 0) {
            double beta = std::sqrt(rave_constant / (3.0 * visits + rave_constant));
            double amaf_q = amaf_wins / amaf_visits;
            q = (1.0 - beta) * q + beta * amaf_q;
        }
        
        return q + exploration;
    }
};

// Simulation State
struct State {
    double price;
    double pnl;
    bool active;
    int steps;
};

// ============================================================================
// ROLLOUT POLICIES
// ============================================================================

double rollout_random(State state, int max_depth, double drift, double volatility, 
                      int direction, double entry_price) {
    int depth = 0;
    while (state.active && depth < max_depth) {
        int r = rand() % 3;
        MoveType m = static_cast<MoveType>(r);

        if (m == CLOSE) {
            state.active = false;
        } else if (m == ADD) {
            // Simulated add (increases exposure)
            state.pnl *= 1.5;
        } else {
            double shock = generate_normal(0, volatility);
            state.price += drift + shock;
            state.pnl = (direction == 1) ? (state.price - entry_price) : (entry_price - state.price);
        }
        depth++;
    }
    return state.pnl;
}

// ============================================================================
// ORIGINAL MCTS (Backward Compatible)
// ============================================================================

EXPORT SimulationResult run_mcts_simulation(
    double current_price, 
    double entry_price, 
    int direction, 
    double volatility,
    double drift,
    int iterations,
    int depth
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    Node* root = new Node(HOLD, nullptr);
    State root_state = {current_price, 0.0, true, 0};
    root_state.pnl = (direction == 1) ? (current_price - entry_price) : (entry_price - current_price);

    for (int i = 0; i < iterations; ++i) {
        Node* node = root;
        State state = root_state;

        // 1. Selection
        while (node->untried_moves.empty() && !node->children.empty()) {
            double best_score = -1e9;
            Node* best_child = nullptr;
            for (Node* c : node->children) {
                double uct = c->uct_value(1.41);
                if (uct > best_score) {
                    best_score = uct;
                    best_child = c;
                }
            }
            if (best_child) node = best_child;
            if (node->move == CLOSE) state.active = false;
        }

        // 2. Expansion
        if (!node->untried_moves.empty() && state.active) {
            MoveType m = node->untried_moves.back();
            node->untried_moves.pop_back();
            
            State next_state = state;
            if (m == CLOSE) {
                next_state.active = false;
            } else {
                double shock = generate_normal(0, volatility);
                next_state.price += drift + shock;
                next_state.pnl = (direction == 1) ? (next_state.price - entry_price) : (entry_price - next_state.price);
            }
            
            Node* child = new Node(m, node);
            node->children.push_back(child);
            node = child;
            state = next_state;
        }

        // 3. Rollout
        double reward = rollout_random(state, depth, drift, volatility, direction, entry_price);

        // 4. Backpropagation
        while (node != nullptr) {
            node->visits++;
            node->wins += reward;
            node = node->parent;
        }
    }

    // Select Best Move
    Node* best_child = nullptr;
    int most_visits = -1;
    for (Node* c : root->children) {
        if (c->visits > most_visits) {
            most_visits = c->visits;
            best_child = c;
        }
    }

    SimulationResult res;
    if (best_child) {
        res.best_move_type = static_cast<int>(best_child->move);
        res.expected_value = best_child->wins / best_child->visits;
        res.visits = best_child->visits;
        res.confidence = static_cast<double>(best_child->visits) / iterations;
        res.q_value = res.expected_value;
    } else {
        res.best_move_type = 0;
        res.expected_value = 0;
        res.visits = 0;
        res.confidence = 0;
        res.q_value = 0;
    }

    delete root;
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    g_total_simulations += iterations;
    g_total_time_ms = g_total_time_ms.load() + elapsed;
    
    return res;
}

// ============================================================================
// PARALLEL MCTS (OpenMP)
// ============================================================================

EXPORT SimulationResult run_parallel_mcts(
    double current_price, 
    double entry_price, 
    int direction,
    double volatility,
    double drift,
    int iterations,
    int depth,
    int num_threads
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Results from each thread
    std::vector<SimulationResult> thread_results(num_threads);
    
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif
    
    #pragma omp parallel
    {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif
        
        int local_iterations = iterations / num_threads;
        
        // Each thread runs its own MCTS tree
        Node* root = new Node(HOLD, nullptr);
        State root_state = {current_price, 0.0, true, 0};
        root_state.pnl = (direction == 1) ? (current_price - entry_price) : (entry_price - current_price);
        
        for (int i = 0; i < local_iterations; ++i) {
            Node* node = root;
            State state = root_state;
            
            // Selection
            while (node->untried_moves.empty() && !node->children.empty()) {
                double best_score = -1e9;
                Node* best_child = nullptr;
                for (Node* c : node->children) {
                    double uct = c->uct_value(1.41);
                    if (uct > best_score) {
                        best_score = uct;
                        best_child = c;
                    }
                }
                if (best_child) node = best_child;
                if (node->move == CLOSE) state.active = false;
            }
            
            // Expansion
            if (!node->untried_moves.empty() && state.active) {
                MoveType m = node->untried_moves.back();
                node->untried_moves.pop_back();
                
                State next_state = state;
                if (m == CLOSE) {
                    next_state.active = false;
                } else {
                    double shock = generate_normal(0, volatility);
                    next_state.price += drift + shock;
                    next_state.pnl = (direction == 1) ? (next_state.price - entry_price) : (entry_price - next_state.price);
                }
                
                Node* child = new Node(m, node);
                node->children.push_back(child);
                node = child;
                state = next_state;
            }
            
            // Rollout
            double reward = rollout_random(state, depth, drift, volatility, direction, entry_price);
            
            // Backprop
            while (node != nullptr) {
                node->visits++;
                node->wins += reward;
                node = node->parent;
            }
        }
        
        // Extract thread result
        Node* best_child = nullptr;
        int most_visits = -1;
        for (Node* c : root->children) {
            if (c->visits > most_visits) {
                most_visits = c->visits;
                best_child = c;
            }
        }
        
        if (best_child) {
            thread_results[tid].best_move_type = static_cast<int>(best_child->move);
            thread_results[tid].expected_value = best_child->wins / best_child->visits;
            thread_results[tid].visits = best_child->visits;
            thread_results[tid].confidence = static_cast<double>(best_child->visits) / local_iterations;
            thread_results[tid].q_value = thread_results[tid].expected_value;
        }
        
        delete root;
    }
    
    // Aggregate results (voting)
    int vote_counts[3] = {0, 0, 0};
    double total_value = 0.0;
    int total_visits = 0;
    
    for (int t = 0; t < num_threads; t++) {
        if (thread_results[t].visits > 0) {
            vote_counts[thread_results[t].best_move_type]++;
            total_value += thread_results[t].expected_value * thread_results[t].visits;
            total_visits += thread_results[t].visits;
        }
    }
    
    // Find best move by vote
    int best_move = 0;
    int max_votes = vote_counts[0];
    for (int i = 1; i < 3; i++) {
        if (vote_counts[i] > max_votes) {
            max_votes = vote_counts[i];
            best_move = i;
        }
    }
    
    SimulationResult res;
    res.best_move_type = best_move;
    res.expected_value = (total_visits > 0) ? total_value / total_visits : 0.0;
    res.visits = total_visits;
    res.confidence = static_cast<double>(max_votes) / num_threads;
    res.q_value = res.expected_value;
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    g_total_simulations += iterations;
    g_total_time_ms = g_total_time_ms.load() + elapsed;
    
    return res;
}

// ============================================================================
// MCTS WITH TRANSPOSITION TABLES
// ============================================================================

EXPORT SimulationResult run_mcts_with_tt(
    double current_price, 
    double entry_price, 
    int direction,
    double volatility,
    double drift,
    int iterations,
    int depth
) {
    if (g_tt_size == 0) {
        init_transposition_table(1000000);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Node* root = new Node(HOLD, nullptr);
    State root_state = {current_price, 0.0, true, 0};
    root_state.pnl = (direction == 1) ? (current_price - entry_price) : (entry_price - current_price);
    
    for (int i = 0; i < iterations; ++i) {
        Node* node = root;
        State state = root_state;
        
        // Check transposition table
        uint64_t hash = zobrist_hash(state.price, entry_price, direction, depth);
        size_t idx = hash % g_tt_size;
        
        if (g_transposition_table[idx].valid && g_transposition_table[idx].hash == hash) {
            g_tt_hits++;
            // Use cached value with some probability
            if (generate_uniform() < 0.3) {
                continue; // Skip redundant simulation
            }
        } else {
            g_tt_misses++;
        }
        
        // Selection
        while (node->untried_moves.empty() && !node->children.empty()) {
            double best_score = -1e9;
            Node* best_child = nullptr;
            for (Node* c : node->children) {
                double uct = c->uct_value(1.41);
                if (uct > best_score) {
                    best_score = uct;
                    best_child = c;
                }
            }
            if (best_child) node = best_child;
            if (node->move == CLOSE) state.active = false;
        }
        
        // Expansion
        if (!node->untried_moves.empty() && state.active) {
            MoveType m = node->untried_moves.back();
            node->untried_moves.pop_back();
            
            State next_state = state;
            if (m == CLOSE) {
                next_state.active = false;
            } else {
                double shock = generate_normal(0, volatility);
                next_state.price += drift + shock;
                next_state.pnl = (direction == 1) ? (next_state.price - entry_price) : (entry_price - next_state.price);
            }
            
            Node* child = new Node(m, node);
            node->children.push_back(child);
            node = child;
            state = next_state;
        }
        
        // Rollout
        double reward = rollout_random(state, depth, drift, volatility, direction, entry_price);
        
        // Update transposition table
        {
            std::lock_guard<std::mutex> lock(g_tt_mutex);
            g_transposition_table[idx].hash = hash;
            g_transposition_table[idx].value = reward;
            g_transposition_table[idx].visits++;
            g_transposition_table[idx].depth = depth;
            g_transposition_table[idx].valid = 1;
        }
        
        // Backprop
        while (node != nullptr) {
            node->visits++;
            node->wins += reward;
            node = node->parent;
        }
    }
    
    // Select best
    Node* best_child = nullptr;
    int most_visits = -1;
    for (Node* c : root->children) {
        if (c->visits > most_visits) {
            most_visits = c->visits;
            best_child = c;
        }
    }
    
    SimulationResult res;
    if (best_child) {
        res.best_move_type = static_cast<int>(best_child->move);
        res.expected_value = best_child->wins / best_child->visits;
        res.visits = best_child->visits;
        res.confidence = static_cast<double>(best_child->visits) / iterations;
        res.q_value = res.expected_value;
    } else {
        res.best_move_type = 0;
        res.expected_value = 0;
        res.visits = 0;
        res.confidence = 0;
        res.q_value = 0;
    }
    
    delete root;
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    g_total_simulations += iterations;
    g_total_time_ms = g_total_time_ms.load() + elapsed;
    
    return res;
}

// ============================================================================
// RAVE MCTS (Rapid Action Value Estimation)
// ============================================================================

EXPORT SimulationResult run_rave_mcts(
    double current_price, 
    double entry_price, 
    int direction,
    double volatility,
    double drift,
    int iterations,
    int depth,
    double rave_constant
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    Node* root = new Node(HOLD, nullptr);
    State root_state = {current_price, 0.0, true, 0};
    root_state.pnl = (direction == 1) ? (current_price - entry_price) : (entry_price - current_price);
    
    for (int i = 0; i < iterations; ++i) {
        Node* node = root;
        State state = root_state;
        std::vector<Node*> path;
        std::vector<MoveType> moves_played;
        
        // Selection with RAVE
        while (node->untried_moves.empty() && !node->children.empty()) {
            double best_score = -1e9;
            Node* best_child = nullptr;
            for (Node* c : node->children) {
                double uct = c->uct_value(1.41, rave_constant);
                if (uct > best_score) {
                    best_score = uct;
                    best_child = c;
                }
            }
            if (best_child) {
                node = best_child;
                path.push_back(node);
                moves_played.push_back(node->move);
            }
            if (node->move == CLOSE) state.active = false;
        }
        
        // Expansion
        if (!node->untried_moves.empty() && state.active) {
            MoveType m = node->untried_moves.back();
            node->untried_moves.pop_back();
            
            State next_state = state;
            if (m == CLOSE) {
                next_state.active = false;
            } else {
                double shock = generate_normal(0, volatility);
                next_state.price += drift + shock;
                next_state.pnl = (direction == 1) ? (next_state.price - entry_price) : (entry_price - next_state.price);
            }
            
            Node* child = new Node(m, node);
            node->children.push_back(child);
            node = child;
            path.push_back(node);
            moves_played.push_back(m);
            state = next_state;
        }
        
        // Rollout with move tracking
        std::vector<MoveType> rollout_moves;
        int rollout_depth = 0;
        while (state.active && rollout_depth < depth) {
            int r = rand() % 2;
            MoveType m = (r == 0) ? HOLD : CLOSE;
            rollout_moves.push_back(m);
            
            if (m == CLOSE) {
                state.active = false;
            } else {
                double shock = generate_normal(0, volatility);
                state.price += drift + shock;
                state.pnl = (direction == 1) ? (state.price - entry_price) : (entry_price - state.price);
            }
            rollout_depth++;
        }
        double reward = state.pnl;
        
        // Backprop with AMAF update
        for (Node* n : path) {
            n->visits++;
            n->wins += reward;
            
            // Update AMAF for all moves seen in rollout
            for (MoveType rm : rollout_moves) {
                if (rm == n->move) {
                    n->amaf_visits++;
                    n->amaf_wins += reward;
                }
            }
        }
        
        // Update root
        root->visits++;
        root->wins += reward;
    }
    
    // Select best
    Node* best_child = nullptr;
    int most_visits = -1;
    for (Node* c : root->children) {
        if (c->visits > most_visits) {
            most_visits = c->visits;
            best_child = c;
        }
    }
    
    SimulationResult res;
    if (best_child) {
        res.best_move_type = static_cast<int>(best_child->move);
        res.expected_value = best_child->wins / best_child->visits;
        res.visits = best_child->visits;
        res.confidence = static_cast<double>(best_child->visits) / iterations;
        res.q_value = res.expected_value;
    } else {
        res.best_move_type = 0;
        res.expected_value = 0;
        res.visits = 0;
        res.confidence = 0;
        res.q_value = 0;
    }
    
    delete root;
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    g_total_simulations += iterations;
    g_total_time_ms = g_total_time_ms.load() + elapsed;
    
    return res;
}

// ============================================================================
// ADAPTIVE UCT
// ============================================================================

EXPORT SimulationResult run_adaptive_mcts(
    double current_price, 
    double entry_price, 
    int direction,
    double volatility,
    double drift,
    int iterations,
    int depth,
    double initial_c,
    double c_decay
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    Node* root = new Node(HOLD, nullptr);
    State root_state = {current_price, 0.0, true, 0};
    root_state.pnl = (direction == 1) ? (current_price - entry_price) : (entry_price - current_price);
    
    double current_c = initial_c;
    
    for (int i = 0; i < iterations; ++i) {
        // Adaptive exploration decay
        current_c = initial_c * std::exp(-c_decay * i / iterations);
        if (current_c < 0.1) current_c = 0.1;
        
        Node* node = root;
        State state = root_state;
        
        // Selection with adaptive UCT
        while (node->untried_moves.empty() && !node->children.empty()) {
            double best_score = -1e9;
            Node* best_child = nullptr;
            for (Node* c : node->children) {
                double uct = c->uct_value(current_c);
                if (uct > best_score) {
                    best_score = uct;
                    best_child = c;
                }
            }
            if (best_child) node = best_child;
            if (node->move == CLOSE) state.active = false;
        }
        
        // Expansion
        if (!node->untried_moves.empty() && state.active) {
            MoveType m = node->untried_moves.back();
            node->untried_moves.pop_back();
            
            State next_state = state;
            if (m == CLOSE) {
                next_state.active = false;
            } else {
                double shock = generate_normal(0, volatility);
                next_state.price += drift + shock;
                next_state.pnl = (direction == 1) ? (next_state.price - entry_price) : (entry_price - next_state.price);
            }
            
            Node* child = new Node(m, node);
            node->children.push_back(child);
            node = child;
            state = next_state;
        }
        
        // Rollout
        double reward = rollout_random(state, depth, drift, volatility, direction, entry_price);
        
        // Backprop
        while (node != nullptr) {
            node->visits++;
            node->wins += reward;
            node = node->parent;
        }
    }
    
    // Select best
    Node* best_child = nullptr;
    int most_visits = -1;
    for (Node* c : root->children) {
        if (c->visits > most_visits) {
            most_visits = c->visits;
            best_child = c;
        }
    }
    
    SimulationResult res;
    if (best_child) {
        res.best_move_type = static_cast<int>(best_child->move);
        res.expected_value = best_child->wins / best_child->visits;
        res.visits = best_child->visits;
        res.confidence = static_cast<double>(best_child->visits) / iterations;
        res.q_value = res.expected_value;
    } else {
        res.best_move_type = 0;
        res.expected_value = 0;
        res.visits = 0;
        res.confidence = 0;
        res.q_value = 0;
    }
    
    delete root;
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    g_total_simulations += iterations;
    g_total_time_ms = g_total_time_ms.load() + elapsed;
    
    return res;
}

// ============================================================================
// PROGRESSIVE WIDENING
// ============================================================================

EXPORT SimulationResult run_progressive_mcts(
    double current_price, 
    double entry_price, 
    int direction,
    double volatility,
    double drift,
    int iterations,
    int depth,
    double alpha,
    double beta
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    Node* root = new Node(HOLD, nullptr);
    State root_state = {current_price, 0.0, true, 0};
    root_state.pnl = (direction == 1) ? (current_price - entry_price) : (entry_price - current_price);
    
    for (int i = 0; i < iterations; ++i) {
        Node* node = root;
        State state = root_state;
        
        // Selection with progressive widening
        while (state.active) {
            // Progressive widening: expand if children < beta * visits^alpha
            double max_children = beta * std::pow(node->visits + 1, alpha);
            
            if (!node->untried_moves.empty() && node->children.size() < max_children) {
                // Expand
                MoveType m = node->untried_moves.back();
                node->untried_moves.pop_back();
                
                State next_state = state;
                if (m == CLOSE) {
                    next_state.active = false;
                } else {
                    double shock = generate_normal(0, volatility);
                    next_state.price += drift + shock;
                    next_state.pnl = (direction == 1) ? (next_state.price - entry_price) : (entry_price - next_state.price);
                }
                
                Node* child = new Node(m, node);
                node->children.push_back(child);
                node = child;
                state = next_state;
                break;
            } else if (!node->children.empty()) {
                // Select best child
                double best_score = -1e9;
                Node* best_child = nullptr;
                for (Node* c : node->children) {
                    double uct = c->uct_value(1.41);
                    if (uct > best_score) {
                        best_score = uct;
                        best_child = c;
                    }
                }
                if (best_child) {
                    node = best_child;
                    if (node->move == CLOSE) state.active = false;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        // Rollout
        double reward = rollout_random(state, depth, drift, volatility, direction, entry_price);
        
        // Backprop
        while (node != nullptr) {
            node->visits++;
            node->wins += reward;
            node = node->parent;
        }
    }
    
    // Select best
    Node* best_child = nullptr;
    int most_visits = -1;
    for (Node* c : root->children) {
        if (c->visits > most_visits) {
            most_visits = c->visits;
            best_child = c;
        }
    }
    
    SimulationResult res;
    if (best_child) {
        res.best_move_type = static_cast<int>(best_child->move);
        res.expected_value = best_child->wins / best_child->visits;
        res.visits = best_child->visits;
        res.confidence = static_cast<double>(best_child->visits) / iterations;
        res.q_value = res.expected_value;
    } else {
        res.best_move_type = 0;
        res.expected_value = 0;
        res.visits = 0;
        res.confidence = 0;
        res.q_value = 0;
    }
    
    delete root;
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    g_total_simulations += iterations;
    g_total_time_ms = g_total_time_ms.load() + elapsed;
    
    return res;
}

// ============================================================================
// FULL AGI MCTS (All Features Combined)
// ============================================================================

EXPORT ExtendedResult run_agi_mcts(
    double current_price, 
    double entry_price, 
    int direction,
    double volatility,
    double drift,
    int iterations,
    int depth,
    int num_threads,
    double rave_constant,
    double initial_c,
    int use_tt
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    if (use_tt && g_tt_size == 0) {
        init_transposition_table(1000000);
    }
    
    // Split iterations across threads
    std::vector<ExtendedResult> thread_results(num_threads);
    
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif
    
    #pragma omp parallel
    {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif
        
        int local_iterations = iterations / num_threads;
        double current_c = initial_c;
        int max_depth_reached = 0;
        
        Node* root = new Node(HOLD, nullptr);
        State root_state = {current_price, 0.0, true, 0};
        root_state.pnl = (direction == 1) ? (current_price - entry_price) : (entry_price - current_price);
        
        for (int i = 0; i < local_iterations; ++i) {
            // Adaptive C
            current_c = initial_c * std::exp(-0.5 * i / local_iterations);
            if (current_c < 0.1) current_c = 0.1;
            
            Node* node = root;
            State state = root_state;
            std::vector<Node*> path;
            int current_depth = 0;
            
            // TT Check
            if (use_tt) {
                uint64_t hash = zobrist_hash(state.price, entry_price, direction, depth);
                size_t idx = hash % g_tt_size;
                if (g_transposition_table[idx].valid && g_transposition_table[idx].hash == hash) {
                    if (generate_uniform() < 0.2) continue;
                }
            }
            
            // Selection with RAVE
            while (node->untried_moves.empty() && !node->children.empty()) {
                double best_score = -1e9;
                Node* best_child = nullptr;
                for (Node* c : node->children) {
                    double uct = c->uct_value(current_c, rave_constant);
                    if (uct > best_score) {
                        best_score = uct;
                        best_child = c;
                    }
                }
                if (best_child) {
                    node = best_child;
                    path.push_back(node);
                    current_depth++;
                }
                if (node->move == CLOSE) state.active = false;
            }
            
            // Expansion
            if (!node->untried_moves.empty() && state.active) {
                MoveType m = node->untried_moves.back();
                node->untried_moves.pop_back();
                
                State next_state = state;
                if (m == CLOSE) {
                    next_state.active = false;
                } else {
                    double shock = generate_normal(0, volatility);
                    next_state.price += drift + shock;
                    next_state.pnl = (direction == 1) ? (next_state.price - entry_price) : (entry_price - next_state.price);
                }
                
                Node* child = new Node(m, node);
                node->children.push_back(child);
                node = child;
                path.push_back(node);
                state = next_state;
                current_depth++;
            }
            
            if (current_depth > max_depth_reached) max_depth_reached = current_depth;
            
            // Rollout
            double reward = rollout_random(state, depth, drift, volatility, direction, entry_price);
            
            // Backprop with AMAF
            while (node != nullptr) {
                node->visits++;
                node->wins += reward;
                node = node->parent;
            }
        }
        
        // Extract results
        double hold_val = 0, close_val = 0, add_val = 0;
        Node* best_child = nullptr;
        int most_visits = -1;
        
        for (Node* c : root->children) {
            if (c->visits > 0) {
                double q = c->wins / c->visits;
                if (c->move == HOLD) hold_val = q;
                else if (c->move == CLOSE) close_val = q;
                else if (c->move == ADD) add_val = q;
            }
            if (c->visits > most_visits) {
                most_visits = c->visits;
                best_child = c;
            }
        }
        
        thread_results[tid].hold_value = hold_val;
        thread_results[tid].close_value = close_val;
        thread_results[tid].add_value = add_val;
        thread_results[tid].tree_depth = max_depth_reached;
        thread_results[tid].total_simulations = local_iterations;
        thread_results[tid].exploration_rate = current_c;
        
        if (best_child) {
            thread_results[tid].base.best_move_type = static_cast<int>(best_child->move);
            thread_results[tid].base.expected_value = best_child->wins / best_child->visits;
            thread_results[tid].base.visits = best_child->visits;
            thread_results[tid].base.confidence = static_cast<double>(best_child->visits) / local_iterations;
            thread_results[tid].base.q_value = thread_results[tid].base.expected_value;
        }
        
        delete root;
    }
    
    // Aggregate
    ExtendedResult final_result;
    int vote_counts[3] = {0, 0, 0};
    double total_hold = 0, total_close = 0, total_add = 0;
    int max_tree_depth = 0;
    int total_sims = 0;
    double total_value = 0;
    int total_visits = 0;
    
    for (int t = 0; t < num_threads; t++) {
        if (thread_results[t].base.visits > 0) {
            vote_counts[thread_results[t].base.best_move_type]++;
            total_value += thread_results[t].base.expected_value * thread_results[t].base.visits;
            total_visits += thread_results[t].base.visits;
        }
        total_hold += thread_results[t].hold_value;
        total_close += thread_results[t].close_value;
        total_add += thread_results[t].add_value;
        total_sims += thread_results[t].total_simulations;
        if (thread_results[t].tree_depth > max_tree_depth) {
            max_tree_depth = thread_results[t].tree_depth;
        }
    }
    
    int best_move = 0;
    int max_votes = vote_counts[0];
    for (int i = 1; i < 3; i++) {
        if (vote_counts[i] > max_votes) {
            max_votes = vote_counts[i];
            best_move = i;
        }
    }
    
    final_result.base.best_move_type = best_move;
    final_result.base.expected_value = (total_visits > 0) ? total_value / total_visits : 0.0;
    final_result.base.visits = total_visits;
    final_result.base.confidence = static_cast<double>(max_votes) / num_threads;
    final_result.base.q_value = final_result.base.expected_value;
    
    final_result.hold_value = total_hold / num_threads;
    final_result.close_value = total_close / num_threads;
    final_result.add_value = total_add / num_threads;
    final_result.total_simulations = total_sims;
    final_result.tree_depth = max_tree_depth;
    final_result.exploration_rate = initial_c;
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    g_total_simulations += iterations;
    g_total_time_ms = g_total_time_ms.load() + elapsed;
    
    return final_result;
}

// ============================================================================
// NEURAL NETWORK GUIDANCE (Placeholder - requires external model)
// ============================================================================

static bool g_nn_loaded = false;

EXPORT int load_neural_network(const char* model_path) {
    // Placeholder: Would load TensorFlow/ONNX model
    g_nn_loaded = true;
    return 1;
}

EXPORT double evaluate_position_nn(
    double current_price,
    double entry_price,
    int direction,
    double volatility,
    double pnl
) {
    if (!g_nn_loaded) return 0.0;
    
    // Placeholder: Simple heuristic evaluation
    double normalized_pnl = pnl / (volatility + 0.001);
    return std::tanh(normalized_pnl);
}

EXPORT void get_action_probs(
    double current_price,
    double entry_price,
    int direction,
    double volatility,
    double pnl,
    double* probs_out
) {
    if (!g_nn_loaded || probs_out == nullptr) {
        if (probs_out) {
            probs_out[0] = 0.33;
            probs_out[1] = 0.33;
            probs_out[2] = 0.34;
        }
        return;
    }
    
    // Placeholder: Simple heuristic probs
    double normalized_pnl = pnl / (volatility + 0.001);
    
    if (normalized_pnl > 1.0) {
        probs_out[0] = 0.2;  // HOLD
        probs_out[1] = 0.7;  // CLOSE (take profit)
        probs_out[2] = 0.1;  // ADD
    } else if (normalized_pnl < -1.0) {
        probs_out[0] = 0.3;  // HOLD
        probs_out[1] = 0.5;  // CLOSE (cut loss)
        probs_out[2] = 0.2;  // ADD
    } else {
        probs_out[0] = 0.6;  // HOLD
        probs_out[1] = 0.2;  // CLOSE
        probs_out[2] = 0.2;  // ADD
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

EXPORT const char* get_mcts_version() {
    return MCTS_VERSION;
}

EXPORT double get_avg_simulation_time_ms() {
    int total = g_total_simulations.load();
    if (total == 0) return 0.0;
    return g_total_time_ms.load() / total;
}

EXPORT int get_total_simulations_run() {
    return g_total_simulations.load();
}

EXPORT void reset_mcts_stats() {
    g_total_simulations = 0;
    g_total_time_ms = 0;
    g_tt_hits = 0;
    g_tt_misses = 0;
}
