
#ifndef PHYSICS_H
#define PHYSICS_H

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

extern "C" {
    
    // --- LAPLACE DEMON ---
    
    struct TrajectoryResult {
        double terminal_price;
        double max_deviation;
        int steps_taken;
        double total_distance;
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

    // --- RIEMANN GEOMETRY ---

    EXPORT double calculate_sectional_curvature(
        double* prices,
        int length,
        int window_size
    );

}

#endif
