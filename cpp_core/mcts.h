
#ifndef MCTS_H
#define MCTS_H

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

extern "C" {
    // Structure to represent a Simulation Result
    struct SimulationResult {
        int best_move_type; // 0=HOLD, 1=CLOSE, 2=ADD
        double expected_value;
        int visits;
    };

    // Main Entry Point
    // drift: Expected price change per step (from Alpha modules)
    // volatility: Price standard deviation per step
    // current_pnl: Current unrealized PnL
    // iterations: Number of MC simulations to run
    // depth: How many steps forward to simulate
    EXPORT SimulationResult run_mcts_simulation(
        double current_price, 
        double entry_price, 
        int direction, // 1=LONG, -1=SHORT
        double volatility,
        double drift,
        int iterations,
        int depth
    );
}

#endif
