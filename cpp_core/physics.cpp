
#include "physics.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

// --- LAPLACE IMPLEMENTATION ---

EXPORT TrajectoryResult simulate_trajectory(
    double start_price,
    double start_velocity,
    double start_accel,
    double mass,
    double friction_coeff,
    double dt,
    int max_steps
) {
    double p = start_price;
    double v = start_velocity;
    double a = start_accel;
    
    double max_dev = 0.0;
    double total_dist = 0.0;
    int step = 0;
    
    for (step = 0; step < max_steps; ++step) {
        // Crash protection
        if (v > 1000.0) v = 1000.0;
        if (v < -1000.0) v = -1000.0;
        
        // Friction Force
        // F_friction = -mu * v (Linear/Stokes) or -mu * v^2 * sign(v)
        // Using hybrid model from Python code
        double f_friction = 0.0;
        if (std::abs(v) < 10.0) {
            f_friction = -1.0 * friction_coeff * v;
        } else {
             f_friction = -1.0 * friction_coeff * std::pow(std::abs(v), 1.5) * (v > 0 ? 1.0 : -1.0);
        }
        
        // Decay initial acceleration (impluse fade)
        a *= 0.9;
        
        // Newton's 2nd Law: F = ma -> a_eff = F/m + a_initial
        double a_eff = a + (f_friction / mass);
        
        // Basic Euler Integration
        v += a_eff * dt;
        p += v * dt;
        
        double dev = std::abs(p - start_price);
        if (dev > max_dev) max_dev = dev;
        
        total_dist += std::abs(v * dt);
        
        // Terminal Condition (Stopped)
        if (std::abs(v) < 0.001 * start_price) {
            break;
        }
    }
    
    TrajectoryResult res;
    res.terminal_price = p;
    res.max_deviation = max_dev;
    res.steps_taken = step;
    res.total_distance = total_dist;
    
    return res;
}

// --- RIEMANN IMPLEMENTATION ---

// Simple helper to calculate discrete curvature roughly analogous to 
// 2nd derivative normalized by arc length (Geodesic deviation)
EXPORT double calculate_sectional_curvature(
    double* prices,
    int length,
    int window_size
) {
    if (length < window_size || window_size < 3) return 0.0;
    
    // We are looking for the "Curvature" K.
    // In differential geometry, K deviates parallel lines.
    // In finance, K deviates linear trends.
    // K ~ (d^2y/dx^2) / (1 + (dy/dx)^2)^(3/2)
    
    // We calculate mean K over the recent window
    double total_k = 0.0;
    
    // Iterate backwards from end
    for (int i = length - 1; i >= length - window_size + 1; --i) {
        double p_now = prices[i];
        double p_prev = prices[i-1];
        double p_prev2 = prices[i-2];
        
        // First Deriv (Velocity)
        double dy_dx = p_now - p_prev;
        
        // Second Deriv (Acceleration)
        double d2y_dx2 = (p_now - p_prev) - (p_prev - p_prev2);
        
        // Curvature Formula
        double denominator = std::pow(1.0 + dy_dx*dy_dx, 1.5);
        if (denominator == 0) denominator = 0.0001;
        
        double k = d2y_dx2 / denominator;
        total_k += k;
    }
    
    return total_k / (window_size - 2);
}
