
#ifndef MCTS_H
#define MCTS_H

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

#include <stdint.h>

extern "C" {
    // ============================================================================
    // MCTS ULTRA-ADVANCED - AGI Core
    // Features: Parallel MCTS, Transposition Tables, RAVE, Adaptive UCT
    // ============================================================================

    // Basic Simulation Result
    struct SimulationResult {
        int best_move_type;      // 0=HOLD, 1=CLOSE, 2=ADD
        double expected_value;   // Expected PnL
        int visits;              // Number of visits to best node
        double confidence;       // Confidence in decision [0,1]
        double q_value;          // Q-value of best action
    };

    // Extended Result with statistics
    struct ExtendedResult {
        SimulationResult base;
        double hold_value;       // Q-value for HOLD
        double close_value;      // Q-value for CLOSE
        double add_value;        // Q-value for ADD
        int total_simulations;   // Total simulations run
        int tree_depth;          // Maximum depth reached
        double exploration_rate; // Current UCT exploration rate
    };

    // Transposition Table Entry
    struct TTEntry {
        uint64_t hash;           // Zobrist hash of state
        double value;            // Cached value
        int visits;              // Visit count
        int depth;               // Search depth
        int8_t valid;            // Entry validity flag
    };

    // RAVE Statistics
    struct RAVEStats {
        double amaf_value;       // All-Moves-As-First value
        int amaf_visits;         // AMAF visit count
        double rave_weight;      // RAVE weighting factor
    };

    // ============================================================================
    // BASIC MCTS (Original API - Backward Compatible)
    // ============================================================================
    
    EXPORT SimulationResult run_mcts_simulation(
        double current_price, 
        double entry_price, 
        int direction,
        double volatility,
        double drift,
        int iterations,
        int depth
    );

    // ============================================================================
    // PARALLEL MCTS (OpenMP Multi-threaded)
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
    );

    // ============================================================================
    // MCTS WITH TRANSPOSITION TABLES
    // ============================================================================
    
    // Initialize transposition table with given size (number of entries)
    EXPORT void init_transposition_table(size_t size);
    
    // Clear transposition table
    EXPORT void clear_transposition_table();
    
    // Get transposition table stats
    EXPORT int get_tt_hits();
    EXPORT int get_tt_misses();
    
    // Run MCTS with transposition tables
    EXPORT SimulationResult run_mcts_with_tt(
        double current_price, 
        double entry_price, 
        int direction,
        double volatility,
        double drift,
        int iterations,
        int depth
    );

    // ============================================================================
    // RAVE (Rapid Action Value Estimation)
    // ============================================================================
    
    EXPORT SimulationResult run_rave_mcts(
        double current_price, 
        double entry_price, 
        int direction,
        double volatility,
        double drift,
        int iterations,
        int depth,
        double rave_constant      // Controls RAVE influence (typically 300-3000)
    );

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
        double initial_c,         // Initial UCT constant
        double c_decay            // Decay rate for exploration
    );

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
        double alpha,             // Widening exponent
        double beta               // Widening coefficient
    );

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
        int num_threads,          // Parallelization
        double rave_constant,     // RAVE influence
        double initial_c,         // UCT constant
        int use_tt                // Use transposition table (0/1)
    );

    // ============================================================================
    // NEURAL NETWORK GUIDANCE (AlphaZero-style)
    // ============================================================================
    
    // Load a neural network model for guidance
    EXPORT int load_neural_network(const char* model_path);
    
    // Evaluate position using neural network
    EXPORT double evaluate_position_nn(
        double current_price,
        double entry_price,
        int direction,
        double volatility,
        double pnl
    );
    
    // Get action probabilities from policy network
    EXPORT void get_action_probs(
        double current_price,
        double entry_price,
        int direction,
        double volatility,
        double pnl,
        double* probs_out         // Output: [HOLD, CLOSE, ADD] probabilities
    );

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================
    
    // Get version info
    EXPORT const char* get_mcts_version();
    
    // Get performance stats
    EXPORT double get_avg_simulation_time_ms();
    EXPORT int get_total_simulations_run();
    
    // Reset all statistics
    EXPORT void reset_mcts_stats();
}

#endif
