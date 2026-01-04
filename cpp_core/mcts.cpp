
#include "mcts.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <ctime>

// Simple Box-Muller for Normal Distribution (Faster than std::normal_distribution for some compilers)
// Or just use std for readability.
double generate_normal(double mean, double stddev, std::mt19937& gen) {
    std::normal_distribution<> d(mean, stddev);
    return d(gen);
}

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

    Node(MoveType m, Node* p) : move(m), parent(p), wins(0.0), visits(0) {
        untried_moves = {HOLD, CLOSE}; // Simplified Action Space
    }

    ~Node() {
        for (Node* c : children) delete c;
    }
};

// Simulation State
struct State {
    double price;
    double pnl;
    bool active;
};

// Rollout Policy
double rollout(State state, int max_depth, double drift, double volatility, int direction, double entry_price, std::mt19937& gen) {
    int depth = 0;
    while (state.active && depth < max_depth) {
        // Random Move
        int r = rand() % 2; // 0 or 1
        MoveType m = (r == 0) ? HOLD : CLOSE;

        if (m == CLOSE) {
            state.active = false;
        } else {
            // Price Walk
            double shock = generate_normal(0, volatility, gen);
            state.price += drift + shock;
            state.pnl = (direction == 1) ? (state.price - entry_price) : (entry_price - state.price);
        }
        depth++;
    }
    return state.pnl; // Utility
}

EXPORT SimulationResult run_mcts_simulation(
    double current_price, 
    double entry_price, 
    int direction, 
    double volatility,
    double drift,
    int iterations,
    int depth
) {
    std::mt19937 gen((unsigned int)time(0));
    
    Node* root = new Node(HOLD, nullptr);
    State root_state = {current_price, 0.0, true};
    root_state.pnl = (direction == 1) ? (current_price - entry_price) : (entry_price - current_price);

    for (int i = 0; i < iterations; ++i) {
        Node* node = root;
        State state = root_state;

        // 1. Selection
        while (node->untried_moves.empty() && !node->children.empty()) {
            // UCT
            double best_score = -1e9;
            Node* best_child = nullptr;
            for (Node* c : node->children) {
                double uct = (c->wins / c->visits) + 1.41 * std::sqrt(std::log(node->visits) / c->visits);
                if (uct > best_score) {
                    best_score = uct;
                    best_child = c;
                }
            }
            if (best_child) node = best_child;
            
            // Advance State for Selection (Ideally we track this better, but simplified for speed)
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
                 // First step of simulation from this node
                 double shock = generate_normal(0, volatility, gen);
                 next_state.price += drift + shock;
                 next_state.pnl = (direction == 1) ? (next_state.price - entry_price) : (entry_price - next_state.price);
            }
            
            Node* child = new Node(m, node);
            node->children.push_back(child);
            node = child;
            state = next_state;
        }

        // 3. Rollout
        double reward = rollout(state, depth, drift, volatility, direction, entry_price, gen);

        // 4. Backprop
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
        res.best_move_type = (best_child->move == CLOSE) ? 1 : 0;
        res.expected_value = best_child->wins / best_child->visits;
        res.visits = best_child->visits;
    } else {
        res.best_move_type = 0; // Hold default
        res.expected_value = 0;
        res.visits = 0;
    }

    delete root;
    return res;
}
