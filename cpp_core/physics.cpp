
#include "physics.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// GLOBAL STATE
// ============================================================================

static const char* PHYSICS_VERSION = "2.0.0-AGI-Ultra";
static thread_local std::mt19937 tl_gen(std::random_device{}());

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static double generate_uniform() {
    std::uniform_real_distribution<> d(0.0, 1.0);
    return d(tl_gen);
}

static double generate_normal(double mean, double stddev) {
    std::normal_distribution<> d(mean, stddev);
    return d(tl_gen);
}

// ============================================================================
// ORIGINAL LAPLACE DEMON
// ============================================================================

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
    double energy = 0.5 * mass * v * v;
    int step = 0;
    
    for (step = 0; step < max_steps; ++step) {
        if (v > 1000.0) v = 1000.0;
        if (v < -1000.0) v = -1000.0;
        
        double f_friction = 0.0;
        if (std::abs(v) < 10.0) {
            f_friction = -1.0 * friction_coeff * v;
        } else {
            f_friction = -1.0 * friction_coeff * std::pow(std::abs(v), 1.5) * (v > 0 ? 1.0 : -1.0);
        }
        
        a *= 0.9;
        double a_eff = a + (f_friction / mass);
        
        v += a_eff * dt;
        p += v * dt;
        
        double dev = std::abs(p - start_price);
        if (dev > max_dev) max_dev = dev;
        
        total_dist += std::abs(v * dt);
        
        if (std::abs(v) < 0.001 * start_price) {
            break;
        }
    }
    
    TrajectoryResult res;
    res.terminal_price = p;
    res.max_deviation = max_dev;
    res.steps_taken = step;
    res.total_distance = total_dist;
    res.final_velocity = v;
    res.energy = 0.5 * mass * v * v;
    
    return res;
}

// ============================================================================
// ORIGINAL RIEMANN GEOMETRY
// ============================================================================

EXPORT double calculate_sectional_curvature(
    double* prices,
    int length,
    int window_size
) {
    if (length < window_size || window_size < 3) return 0.0;
    
    double total_k = 0.0;
    
    for (int i = length - 1; i >= length - window_size + 1; --i) {
        double p_now = prices[i];
        double p_prev = prices[i-1];
        double p_prev2 = prices[i-2];
        
        double dy_dx = p_now - p_prev;
        double d2y_dx2 = (p_now - p_prev) - (p_prev - p_prev2);
        
        double denominator = std::pow(1.0 + dy_dx*dy_dx, 1.5);
        if (denominator == 0) denominator = 0.0001;
        
        double k = d2y_dx2 / denominator;
        total_k += k;
    }
    
    return total_k / (window_size - 2);
}

// ============================================================================
// MULTI-BODY PHYSICS
// ============================================================================

EXPORT MultiBodyResult* simulate_multi_body(
    double* prices,
    double* velocities,
    double* masses,
    int num_bodies,
    double dt,
    int max_steps,
    double interaction_strength
) {
    MultiBodyResult* result = new MultiBodyResult();
    result->terminal_prices = new double[num_bodies];
    result->velocities = new double[num_bodies];
    
    std::vector<double> pos(prices, prices + num_bodies);
    std::vector<double> vel(velocities, velocities + num_bodies);
    std::vector<double> acc(num_bodies, 0.0);
    
    for (int step = 0; step < max_steps; step++) {
        // Calculate forces (gravitational-like attraction)
        std::fill(acc.begin(), acc.end(), 0.0);
        
        for (int i = 0; i < num_bodies; i++) {
            for (int j = i + 1; j < num_bodies; j++) {
                double diff = pos[j] - pos[i];
                double dist = std::abs(diff) + 0.001;
                double force = interaction_strength * masses[i] * masses[j] / (dist * dist);
                double direction = (diff > 0) ? 1.0 : -1.0;
                
                acc[i] += force * direction / masses[i];
                acc[j] -= force * direction / masses[j];
            }
        }
        
        // Verlet integration
        for (int i = 0; i < num_bodies; i++) {
            vel[i] += acc[i] * dt;
            pos[i] += vel[i] * dt;
        }
        
        result->steps_taken = step + 1;
    }
    
    // Calculate final values
    double total_mass = 0.0;
    double com = 0.0;
    double momentum = 0.0;
    double energy = 0.0;
    
    for (int i = 0; i < num_bodies; i++) {
        result->terminal_prices[i] = pos[i];
        result->velocities[i] = vel[i];
        
        total_mass += masses[i];
        com += pos[i] * masses[i];
        momentum += masses[i] * vel[i];
        energy += 0.5 * masses[i] * vel[i] * vel[i];
    }
    
    result->center_of_mass = com / total_mass;
    result->total_momentum = momentum;
    result->total_energy = energy;
    
    return result;
}

EXPORT void free_multi_body_result(MultiBodyResult* result) {
    if (result) {
        delete[] result->terminal_prices;
        delete[] result->velocities;
        delete result;
    }
}

// ============================================================================
// QUANTUM MECHANICS
// ============================================================================

EXPORT double calculate_tunneling_probability(
    double barrier_height,
    double particle_energy,
    double barrier_width
) {
    if (particle_energy >= barrier_height) return 1.0;
    
    // WKB approximation: T ≈ exp(-2 * κ * L)
    // κ = sqrt(2m(V-E)) / ħ, simplified
    double delta_v = barrier_height - particle_energy;
    double kappa = std::sqrt(2.0 * delta_v);
    double transmission = std::exp(-2.0 * kappa * barrier_width);
    
    return std::min(1.0, std::max(0.0, transmission));
}

EXPORT QuantumState* create_quantum_state(
    double* prices,
    double* probabilities,
    int num_prices
) {
    QuantumState* state = new QuantumState();
    state->num_states = num_prices;
    state->amplitudes = new double[num_prices];
    state->phases = new double[num_prices];
    
    // Convert probabilities to amplitudes
    double norm = 0.0;
    for (int i = 0; i < num_prices; i++) {
        state->amplitudes[i] = std::sqrt(probabilities[i]);
        state->phases[i] = generate_uniform() * 2.0 * M_PI;
        norm += state->amplitudes[i] * state->amplitudes[i];
    }
    
    // Normalize
    norm = std::sqrt(norm);
    for (int i = 0; i < num_prices; i++) {
        state->amplitudes[i] /= norm;
    }
    
    // Coherence based on phase variance
    double mean_phase = 0.0;
    for (int i = 0; i < num_prices; i++) {
        mean_phase += state->phases[i];
    }
    mean_phase /= num_prices;
    
    double phase_variance = 0.0;
    for (int i = 0; i < num_prices; i++) {
        double diff = state->phases[i] - mean_phase;
        phase_variance += diff * diff;
    }
    phase_variance /= num_prices;
    
    state->coherence = std::exp(-phase_variance);
    
    return state;
}

EXPORT double quantum_expected_value(QuantumState* state) {
    if (!state) return 0.0;
    
    double expected = 0.0;
    for (int i = 0; i < state->num_states; i++) {
        double prob = state->amplitudes[i] * state->amplitudes[i];
        expected += i * prob;  // Using index as value proxy
    }
    return expected;
}

EXPORT void collapse_quantum_state(QuantumState* state, double measurement) {
    if (!state) return;
    
    // Find closest eigenstate to measurement
    int closest = 0;
    for (int i = 1; i < state->num_states; i++) {
        if (std::abs(i - measurement) < std::abs(closest - measurement)) {
            closest = i;
        }
    }
    
    // Collapse to that state
    for (int i = 0; i < state->num_states; i++) {
        state->amplitudes[i] = (i == closest) ? 1.0 : 0.0;
    }
    state->coherence = 0.0;
}

EXPORT void free_quantum_state(QuantumState* state) {
    if (state) {
        delete[] state->amplitudes;
        delete[] state->phases;
        delete state;
    }
}

EXPORT void calculate_wave_function(
    double* prices,
    int length,
    double* psi_real,
    double* psi_imag,
    int output_size
) {
    // Gaussian wave packet
    double mean = 0.0, variance = 0.0;
    for (int i = 0; i < length; i++) mean += prices[i];
    mean /= length;
    
    for (int i = 0; i < length; i++) {
        double diff = prices[i] - mean;
        variance += diff * diff;
    }
    variance /= length;
    double sigma = std::sqrt(variance);
    
    for (int i = 0; i < output_size; i++) {
        double x = -3.0 * sigma + (6.0 * sigma * i / output_size);
        double gaussian = std::exp(-x * x / (2.0 * sigma * sigma));
        double k = 2.0 * M_PI / sigma;  // Wave number
        
        psi_real[i] = gaussian * std::cos(k * x);
        psi_imag[i] = gaussian * std::sin(k * x);
    }
}

// ============================================================================
// RELATIVITY EFFECTS
// ============================================================================

EXPORT double calculate_time_dilation(double velocity, double c) {
    if (std::abs(velocity) >= c) return 1e9;  // Infinite dilation
    double gamma = 1.0 / std::sqrt(1.0 - (velocity * velocity) / (c * c));
    return gamma;
}

EXPORT double calculate_relativistic_momentum(double mass, double velocity, double c) {
    double gamma = calculate_time_dilation(velocity, c);
    return gamma * mass * velocity;
}

EXPORT TrajectoryResult simulate_relativistic_trajectory(
    double start_price,
    double velocity,
    double mass,
    double c,
    double dt,
    int max_steps
) {
    double p = start_price;
    double v = velocity;
    double max_dev = 0.0;
    double total_dist = 0.0;
    int step = 0;
    
    for (step = 0; step < max_steps; step++) {
        double gamma = calculate_time_dilation(v, c);
        double proper_dt = dt / gamma;  // Time dilation
        
        // Friction with relativistic correction
        double friction = -0.01 * v * gamma;
        double a = friction / (gamma * gamma * gamma * mass);  // Relativistic acceleration
        
        v += a * proper_dt;
        
        // Limit velocity to < c
        if (std::abs(v) > 0.99 * c) v = 0.99 * c * (v > 0 ? 1.0 : -1.0);
        
        p += v * proper_dt;
        
        double dev = std::abs(p - start_price);
        if (dev > max_dev) max_dev = dev;
        total_dist += std::abs(v * proper_dt);
        
        if (std::abs(v) < 0.001 * c) break;
    }
    
    TrajectoryResult res;
    res.terminal_price = p;
    res.max_deviation = max_dev;
    res.steps_taken = step;
    res.total_distance = total_dist;
    res.final_velocity = v;
    res.energy = calculate_relativistic_momentum(mass, v, c) * c;
    
    return res;
}

// ============================================================================
// CHAOS THEORY
// ============================================================================

EXPORT double calculate_lyapunov_exponent(double* prices, int length) {
    if (length < 10) return 0.0;
    
    // Calculate Lyapunov exponent using Wolf's algorithm (simplified)
    double sum_log = 0.0;
    int count = 0;
    double epsilon = 0.01;
    
    for (int i = 0; i < length - 1; i++) {
        // Find nearby trajectory
        int nearest = -1;
        double min_dist = 1e9;
        
        for (int j = 0; j < length; j++) {
            if (std::abs(i - j) > 5) {  // Avoid temporal neighbors
                double dist = std::abs(prices[i] - prices[j]);
                if (dist < min_dist && dist > epsilon * 0.1) {
                    min_dist = dist;
                    nearest = j;
                }
            }
        }
        
        if (nearest >= 0 && nearest + 1 < length && i + 1 < length) {
            double d0 = min_dist;
            double d1 = std::abs(prices[i + 1] - prices[nearest + 1]);
            
            if (d0 > 0 && d1 > 0) {
                sum_log += std::log(d1 / d0);
                count++;
            }
        }
    }
    
    return (count > 0) ? sum_log / count : 0.0;
}

EXPORT LorenzState* simulate_lorenz(
    double x0, double y0, double z0,
    double sigma, double rho, double beta,
    double dt,
    int steps,
    int* output_length
) {
    LorenzState* states = new LorenzState[steps];
    *output_length = steps;
    
    double x = x0, y = y0, z = z0;
    
    for (int i = 0; i < steps; i++) {
        double dx = sigma * (y - x);
        double dy = x * (rho - z) - y;
        double dz = x * y - beta * z;
        
        x += dx * dt;
        y += dy * dt;
        z += dz * dt;
        
        states[i].x = x;
        states[i].y = y;
        states[i].z = z;
    }
    
    return states;
}

EXPORT void free_lorenz_state(LorenzState* states) {
    delete[] states;
}

EXPORT double* calculate_bifurcation_diagram(
    double r_start,
    double r_end,
    int r_steps,
    int iterations_per_r,
    int* output_length
) {
    int discard = 100;
    int keep = iterations_per_r - discard;
    *output_length = r_steps * keep;
    
    double* results = new double[*output_length];
    int idx = 0;
    
    for (int ri = 0; ri < r_steps; ri++) {
        double r = r_start + (r_end - r_start) * ri / r_steps;
        double x = 0.5;
        
        // Discard transients
        for (int i = 0; i < discard; i++) {
            x = r * x * (1.0 - x);
        }
        
        // Keep attractor values
        for (int i = 0; i < keep && idx < *output_length; i++) {
            x = r * x * (1.0 - x);
            results[idx++] = x;
        }
    }
    
    return results;
}

EXPORT double calculate_fractal_dimension(double* prices, int length) {
    if (length < 10) return 1.0;
    
    // Box-counting method (simplified)
    double min_p = prices[0], max_p = prices[0];
    for (int i = 1; i < length; i++) {
        if (prices[i] < min_p) min_p = prices[i];
        if (prices[i] > max_p) max_p = prices[i];
    }
    
    double range = max_p - min_p;
    if (range < 0.001) return 1.0;
    
    std::vector<double> log_eps, log_n;
    
    for (int box_size = 2; box_size <= length / 4; box_size *= 2) {
        int num_boxes = (length + box_size - 1) / box_size;
        int count = 0;
        
        for (int b = 0; b < num_boxes; b++) {
            bool has_point = false;
            for (int i = b * box_size; i < std::min((b + 1) * box_size, length); i++) {
                has_point = true;
                break;
            }
            if (has_point) count++;
        }
        
        if (count > 0) {
            log_eps.push_back(std::log(1.0 / box_size));
            log_n.push_back(std::log(count));
        }
    }
    
    // Linear regression for slope
    if (log_eps.size() < 2) return 1.0;
    
    double sum_xy = 0, sum_x = 0, sum_y = 0, sum_x2 = 0;
    int n = log_eps.size();
    
    for (int i = 0; i < n; i++) {
        sum_x += log_eps[i];
        sum_y += log_n[i];
        sum_xy += log_eps[i] * log_n[i];
        sum_x2 += log_eps[i] * log_eps[i];
    }
    
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    return std::abs(slope);
}

EXPORT double calculate_hurst_exponent(double* prices, int length) {
    if (length < 20) return 0.5;
    
    // R/S analysis
    std::vector<double> returns(length - 1);
    for (int i = 0; i < length - 1; i++) {
        returns[i] = std::log(prices[i + 1] / prices[i]);
    }
    
    double mean = 0.0;
    for (double r : returns) mean += r;
    mean /= returns.size();
    
    double cumsum = 0.0;
    double max_cumsum = -1e9, min_cumsum = 1e9;
    
    for (double r : returns) {
        cumsum += (r - mean);
        if (cumsum > max_cumsum) max_cumsum = cumsum;
        if (cumsum < min_cumsum) min_cumsum = cumsum;
    }
    
    double R = max_cumsum - min_cumsum;
    
    double variance = 0.0;
    for (double r : returns) {
        double diff = r - mean;
        variance += diff * diff;
    }
    double S = std::sqrt(variance / returns.size());
    
    if (S < 1e-10) return 0.5;
    
    double RS = R / S;
    double H = std::log(RS) / std::log(length);
    
    return std::max(0.0, std::min(1.0, H));
}

// ============================================================================
// FLUID DYNAMICS
// ============================================================================

EXPORT FlowField* simulate_order_flow(
    double* order_sizes,
    double* order_prices,
    int num_orders,
    double viscosity,
    double dt,
    int steps
) {
    int grid_size = 32;
    FlowField* field = new FlowField();
    field->width = grid_size;
    field->height = grid_size;
    
    field->velocity_x = new double[grid_size * grid_size];
    field->velocity_y = new double[grid_size * grid_size];
    field->pressure = new double[grid_size * grid_size];
    field->density = new double[grid_size * grid_size];
    
    std::memset(field->velocity_x, 0, grid_size * grid_size * sizeof(double));
    std::memset(field->velocity_y, 0, grid_size * grid_size * sizeof(double));
    std::memset(field->pressure, 0, grid_size * grid_size * sizeof(double));
    std::memset(field->density, 0, grid_size * grid_size * sizeof(double));
    
    // Initialize from orders
    double min_price = order_prices[0], max_price = order_prices[0];
    for (int i = 1; i < num_orders; i++) {
        if (order_prices[i] < min_price) min_price = order_prices[i];
        if (order_prices[i] > max_price) max_price = order_prices[i];
    }
    
    for (int i = 0; i < num_orders; i++) {
        int x = (int)((order_prices[i] - min_price) / (max_price - min_price + 0.001) * (grid_size - 1));
        int y = i * grid_size / num_orders;
        
        x = std::max(0, std::min(grid_size - 1, x));
        y = std::max(0, std::min(grid_size - 1, y));
        
        field->density[y * grid_size + x] += std::abs(order_sizes[i]);
        field->velocity_y[y * grid_size + x] += order_sizes[i];
    }
    
    // Simple diffusion steps
    for (int s = 0; s < steps; s++) {
        for (int y = 1; y < grid_size - 1; y++) {
            for (int x = 1; x < grid_size - 1; x++) {
                int idx = y * grid_size + x;
                
                field->velocity_x[idx] += viscosity * (
                    field->velocity_x[idx - 1] + field->velocity_x[idx + 1] - 
                    2 * field->velocity_x[idx]
                ) * dt;
                
                field->velocity_y[idx] += viscosity * (
                    field->velocity_y[idx - grid_size] + field->velocity_y[idx + grid_size] - 
                    2 * field->velocity_y[idx]
                ) * dt;
            }
        }
    }
    
    return field;
}

EXPORT void free_flow_field(FlowField* field) {
    if (field) {
        delete[] field->velocity_x;
        delete[] field->velocity_y;
        delete[] field->pressure;
        delete[] field->density;
        delete field;
    }
}

EXPORT double calculate_flow_divergence(double* prices, double* volumes, int length) {
    if (length < 2) return 0.0;
    
    double divergence = 0.0;
    for (int i = 1; i < length; i++) {
        double dp = prices[i] - prices[i - 1];
        double dv = volumes[i] - volumes[i - 1];
        divergence += dp * dv;
    }
    
    return divergence / (length - 1);
}

EXPORT double calculate_vorticity(double* prices, int length, int window) {
    if (length < window + 2) return 0.0;
    
    double vorticity = 0.0;
    
    for (int i = window; i < length - 1; i++) {
        double curl = (prices[i + 1] - prices[i]) - (prices[i - window + 1] - prices[i - window]);
        vorticity += curl;
    }
    
    return vorticity / (length - window - 1);
}

// ============================================================================
// FIELD THEORY
// ============================================================================

EXPORT FieldStrength* calculate_field_strength(double* prices, double* volumes, int length) {
    FieldStrength* field = new FieldStrength();
    field->length = length;
    field->field_values = new double[length];
    field->gradient = new double[length];
    field->potential = new double[length];
    
    // Field as price * volume
    for (int i = 0; i < length; i++) {
        field->field_values[i] = prices[i] * volumes[i];
    }
    
    // Gradient (derivative)
    field->gradient[0] = 0;
    for (int i = 1; i < length; i++) {
        field->gradient[i] = field->field_values[i] - field->field_values[i - 1];
    }
    
    // Potential (negative integral)
    field->potential[0] = 0;
    for (int i = 1; i < length; i++) {
        field->potential[i] = field->potential[i - 1] - field->gradient[i];
    }
    
    return field;
}

EXPORT void free_field_strength(FieldStrength* field) {
    if (field) {
        delete[] field->field_values;
        delete[] field->gradient;
        delete[] field->potential;
        delete field;
    }
}

EXPORT double calculate_field_energy(double* prices, double* volumes, int length) {
    double energy = 0.0;
    for (int i = 0; i < length; i++) {
        energy += 0.5 * prices[i] * prices[i] * volumes[i];
    }
    return energy / length;
}

EXPORT void trace_field_lines(
    double* prices,
    double* volumes,
    int length,
    double* field_lines,
    int num_lines
) {
    for (int l = 0; l < num_lines; l++) {
        int start = l * length / num_lines;
        double pos = prices[start];
        
        for (int i = start; i < length; i++) {
            double field = prices[i] * volumes[i] / (volumes[i] + 1.0);
            pos += field * 0.01;
            field_lines[l * length + i] = pos;
        }
    }
}

// ============================================================================
// THERMODYNAMICS
// ============================================================================

EXPORT double calculate_market_entropy(double* prices, int length, int bins) {
    if (length < 2 || bins < 2) return 0.0;
    
    double min_p = prices[0], max_p = prices[0];
    for (int i = 1; i < length; i++) {
        if (prices[i] < min_p) min_p = prices[i];
        if (prices[i] > max_p) max_p = prices[i];
    }
    
    double range = max_p - min_p;
    if (range < 1e-10) return 0.0;
    
    std::vector<int> histogram(bins, 0);
    for (int i = 0; i < length; i++) {
        int bin = (int)((prices[i] - min_p) / range * (bins - 1));
        bin = std::max(0, std::min(bins - 1, bin));
        histogram[bin]++;
    }
    
    double entropy = 0.0;
    for (int b = 0; b < bins; b++) {
        if (histogram[b] > 0) {
            double p = (double)histogram[b] / length;
            entropy -= p * std::log(p);
        }
    }
    
    return entropy;
}

EXPORT double calculate_market_temperature(double* prices, int length) {
    if (length < 2) return 0.0;
    
    double mean = 0.0;
    for (int i = 0; i < length; i++) mean += prices[i];
    mean /= length;
    
    double variance = 0.0;
    for (int i = 0; i < length; i++) {
        double diff = prices[i] - mean;
        variance += diff * diff;
    }
    
    return std::sqrt(variance / length);  // Standard deviation as temperature
}

EXPORT double calculate_free_energy(double* prices, int length, double temperature) {
    double entropy = calculate_market_entropy(prices, length, 20);
    double energy = 0.0;
    
    for (int i = 0; i < length; i++) {
        energy += prices[i] * prices[i];
    }
    energy /= length;
    
    // F = U - TS
    return energy - temperature * entropy;
}

EXPORT int detect_phase_transition(double* prices, int length, double* transition_point) {
    if (length < 20) {
        *transition_point = 0;
        return 0;
    }
    
    // Look for sudden change in variance
    int window = 10;
    double max_var_change = 0.0;
    int transition_idx = -1;
    
    for (int i = window; i < length - window; i++) {
        double var_before = 0.0, var_after = 0.0;
        double mean_before = 0.0, mean_after = 0.0;
        
        for (int j = i - window; j < i; j++) mean_before += prices[j];
        mean_before /= window;
        
        for (int j = i; j < i + window; j++) mean_after += prices[j];
        mean_after /= window;
        
        for (int j = i - window; j < i; j++) {
            double diff = prices[j] - mean_before;
            var_before += diff * diff;
        }
        
        for (int j = i; j < i + window; j++) {
            double diff = prices[j] - mean_after;
            var_after += diff * diff;
        }
        
        double var_change = std::abs(var_after - var_before);
        if (var_change > max_var_change) {
            max_var_change = var_change;
            transition_idx = i;
        }
    }
    
    if (transition_idx >= 0 && max_var_change > 1.0) {
        *transition_point = prices[transition_idx];
        return 1;
    }
    
    *transition_point = 0;
    return 0;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

EXPORT const char* get_physics_version() {
    return PHYSICS_VERSION;
}

EXPORT void reset_physics_stats() {
    // No persistent stats currently
}
