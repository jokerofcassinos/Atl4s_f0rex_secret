
#ifndef PHYSICS_H
#define PHYSICS_H

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

#include <stdint.h>

extern "C" {
    
    // ============================================================================
    // PHYSICS ULTRA-ADVANCED - AGI Core
    // Features: Multi-Body, Quantum, Chaos, Fluid Dynamics, Field Theory
    // ============================================================================

    // --- LAPLACE DEMON (Original) ---
    
    struct TrajectoryResult {
        double terminal_price;
        double max_deviation;
        int steps_taken;
        double total_distance;
        double final_velocity;
        double energy;
    };

    EXPORT TrajectoryResult simulate_trajectory(
        double start_price,
        double start_velocity,
        double start_accel,
        double mass,
        double friction_coeff,
        double dt,
        int max_steps
    );

    // --- RIEMANN GEOMETRY (Original) ---

    EXPORT double calculate_sectional_curvature(
        double* prices,
        int length,
        int window_size
    );

    // ============================================================================
    // MULTI-BODY PHYSICS
    // ============================================================================
    
    struct MultiBodyResult {
        double* terminal_prices;    // Array of final prices
        double* velocities;         // Array of final velocities
        double center_of_mass;      // System center of mass
        double total_momentum;      // Total system momentum
        double total_energy;        // Total system energy
        int steps_taken;
    };

    EXPORT MultiBodyResult* simulate_multi_body(
        double* prices,             // Initial prices array
        double* velocities,         // Initial velocities array
        double* masses,             // Mass of each body
        int num_bodies,
        double dt,
        int max_steps,
        double interaction_strength // Gravitational-like interaction
    );
    
    EXPORT void free_multi_body_result(MultiBodyResult* result);

    // ============================================================================
    // QUANTUM MECHANICS
    // ============================================================================
    
    // Calculate quantum tunneling probability (barrier breakthrough)
    EXPORT double calculate_tunneling_probability(
        double barrier_height,      // Resistance level difference
        double particle_energy,     // Current momentum/energy
        double barrier_width        // Zone width
    );
    
    // Quantum superposition state (price exists in multiple states)
    struct QuantumState {
        double* amplitudes;         // Probability amplitudes
        double* phases;             // Phase angles
        int num_states;
        double coherence;           // Quantum coherence measure
    };
    
    EXPORT QuantumState* create_quantum_state(
        double* prices,
        double* probabilities,
        int num_prices
    );
    
    EXPORT double quantum_expected_value(QuantumState* state);
    EXPORT void collapse_quantum_state(QuantumState* state, double measurement);
    EXPORT void free_quantum_state(QuantumState* state);
    
    // Wave function for price probability distribution
    EXPORT void calculate_wave_function(
        double* prices,
        int length,
        double* psi_real,           // Real part of wave function
        double* psi_imag,           // Imaginary part
        int output_size
    );

    // ============================================================================
    // RELATIVITY EFFECTS
    // ============================================================================
    
    // Time dilation for fast-moving markets
    EXPORT double calculate_time_dilation(
        double velocity,            // Price velocity
        double c                    // "Speed of light" (max market velocity)
    );
    
    // Relativistic momentum
    EXPORT double calculate_relativistic_momentum(
        double mass,
        double velocity,
        double c
    );
    
    // Relativistic trajectory simulation
    EXPORT TrajectoryResult simulate_relativistic_trajectory(
        double start_price,
        double velocity,
        double mass,
        double c,                   // Speed limit
        double dt,
        int max_steps
    );

    // ============================================================================
    // CHAOS THEORY
    // ============================================================================
    
    // Lyapunov exponent (sensitivity to initial conditions)
    EXPORT double calculate_lyapunov_exponent(
        double* prices,
        int length
    );
    
    // Lorenz attractor for market dynamics
    struct LorenzState {
        double x, y, z;
    };
    
    EXPORT LorenzState* simulate_lorenz(
        double x0, double y0, double z0,
        double sigma, double rho, double beta,
        double dt,
        int steps,
        int* output_length
    );
    
    EXPORT void free_lorenz_state(LorenzState* states);
    
    // Bifurcation analysis
    EXPORT double* calculate_bifurcation_diagram(
        double r_start,
        double r_end,
        int r_steps,
        int iterations_per_r,
        int* output_length
    );
    
    // Fractal dimension (box-counting)
    EXPORT double calculate_fractal_dimension(
        double* prices,
        int length
    );
    
    // Hurst exponent (long-term memory)
    EXPORT double calculate_hurst_exponent(
        double* prices,
        int length
    );

    // ============================================================================
    // FLUID DYNAMICS (Order Flow)
    // ============================================================================
    
    struct FlowField {
        double* velocity_x;         // Horizontal flow component
        double* velocity_y;         // Vertical flow component
        double* pressure;           // Pressure field
        double* density;            // Density field
        int width;
        int height;
    };
    
    EXPORT FlowField* simulate_order_flow(
        double* order_sizes,        // Order sizes
        double* order_prices,       // Order prices
        int num_orders,
        double viscosity,           // Market viscosity
        double dt,
        int steps
    );
    
    EXPORT void free_flow_field(FlowField* field);
    
    // Navier-Stokes inspired market flow
    EXPORT double calculate_flow_divergence(
        double* prices,
        double* volumes,
        int length
    );
    
    // Vorticity (rotational market patterns)
    EXPORT double calculate_vorticity(
        double* prices,
        int length,
        int window
    );

    // ============================================================================
    // FIELD THEORY
    // ============================================================================
    
    struct FieldStrength {
        double* field_values;       // Field strength at each point
        double* gradient;           // Field gradient
        double* potential;          // Potential energy
        int length;
    };
    
    EXPORT FieldStrength* calculate_field_strength(
        double* prices,
        double* volumes,
        int length
    );
    
    EXPORT void free_field_strength(FieldStrength* field);
    
    // Electromagnetic-like market field
    EXPORT double calculate_field_energy(
        double* prices,
        double* volumes,
        int length
    );
    
    // Field line calculation
    EXPORT void trace_field_lines(
        double* prices,
        double* volumes,
        int length,
        double* field_lines,
        int num_lines
    );

    // ============================================================================
    // THERMODYNAMICS
    // ============================================================================
    
    // Market entropy
    EXPORT double calculate_market_entropy(
        double* prices,
        int length,
        int bins
    );
    
    // Market temperature (volatility analog)
    EXPORT double calculate_market_temperature(
        double* prices,
        int length
    );
    
    // Free energy calculation
    EXPORT double calculate_free_energy(
        double* prices,
        int length,
        double temperature
    );
    
    // Phase transition detection
    EXPORT int detect_phase_transition(
        double* prices,
        int length,
        double* transition_point
    );

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================
    
    EXPORT const char* get_physics_version();
    EXPORT void reset_physics_stats();
}

#endif
