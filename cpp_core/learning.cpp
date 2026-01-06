
#include "learning.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// GLOBAL STATE
// ============================================================================

static const char* LEARNING_VERSION = "1.0.0-AGI";
static std::mt19937 g_gen(42);

EXPORT void set_learning_seed(unsigned int seed) {
    g_gen.seed(seed);
}

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

static float relu(float x) { return x > 0 ? x : 0; }
static float relu_deriv(float x) { return x > 0 ? 1.0f : 0.0f; }

static float sigmoid(float x) {
    if (x > 20) return 1.0f;
    if (x < -20) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}
static float sigmoid_deriv(float x) { float s = sigmoid(x); return s * (1 - s); }

static float tanh_act(float x) { return std::tanh(x); }
static float tanh_deriv(float x) { float t = std::tanh(x); return 1 - t * t; }

// ============================================================================
// NEURAL NETWORK
// ============================================================================

EXPORT NeuralNetwork* create_neural_network(
    const int* layer_sizes,
    const int* activations,
    int num_layers,
    float learning_rate
) {
    NeuralNetwork* nn = new NeuralNetwork();
    nn->num_layers = num_layers - 1;  // Number of weight matrices
    nn->learning_rate = learning_rate;
    nn->momentum = 0.9f;
    
    nn->layers = new DenseLayer[nn->num_layers];
    
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    int total_outputs = 0;
    for (int i = 0; i < nn->num_layers; i++) {
        DenseLayer* layer = &nn->layers[i];
        layer->input_size = layer_sizes[i];
        layer->output_size = layer_sizes[i + 1];
        layer->activation = (ActivationType)activations[i];
        
        int num_weights = layer->input_size * layer->output_size;
        layer->weights = new float[num_weights];
        layer->biases = new float[layer->output_size];
        layer->weight_gradients = new float[num_weights];
        layer->bias_gradients = new float[layer->output_size];
        
        // Xavier initialization
        float std = std::sqrt(2.0f / (layer->input_size + layer->output_size));
        std::normal_distribution<float> init_dist(0.0f, std);
        
        for (int j = 0; j < num_weights; j++) {
            layer->weights[j] = init_dist(g_gen);
            layer->weight_gradients[j] = 0.0f;
        }
        
        for (int j = 0; j < layer->output_size; j++) {
            layer->biases[j] = 0.0f;
            layer->bias_gradients[j] = 0.0f;
        }
        
        total_outputs += layer->output_size;
    }
    
    nn->layer_outputs = new float[total_outputs];
    nn->layer_deltas = new float[total_outputs];
    
    return nn;
}

EXPORT void free_neural_network(NeuralNetwork* nn) {
    if (nn) {
        for (int i = 0; i < nn->num_layers; i++) {
            delete[] nn->layers[i].weights;
            delete[] nn->layers[i].biases;
            delete[] nn->layers[i].weight_gradients;
            delete[] nn->layers[i].bias_gradients;
        }
        delete[] nn->layers;
        delete[] nn->layer_outputs;
        delete[] nn->layer_deltas;
        delete nn;
    }
}

EXPORT void nn_forward(NeuralNetwork* nn, const float* input, float* output) {
    const float* current_input = input;
    int output_offset = 0;
    
    for (int l = 0; l < nn->num_layers; l++) {
        DenseLayer* layer = &nn->layers[l];
        float* layer_out = nn->layer_outputs + output_offset;
        
        // Matrix multiply + bias
        for (int j = 0; j < layer->output_size; j++) {
            float sum = layer->biases[j];
            for (int i = 0; i < layer->input_size; i++) {
                sum += current_input[i] * layer->weights[i * layer->output_size + j];
            }
            
            // Activation
            switch (layer->activation) {
                case ACTIVATION_RELU: layer_out[j] = relu(sum); break;
                case ACTIVATION_SIGMOID: layer_out[j] = sigmoid(sum); break;
                case ACTIVATION_TANH: layer_out[j] = tanh_act(sum); break;
                default: layer_out[j] = sum; break;
            }
        }
        
        current_input = layer_out;
        output_offset += layer->output_size;
    }
    
    // Copy final output
    DenseLayer* last = &nn->layers[nn->num_layers - 1];
    memcpy(output, nn->layer_outputs + output_offset - last->output_size, 
           last->output_size * sizeof(float));
}

EXPORT float nn_backward(NeuralNetwork* nn, const float* input, const float* target) {
    // Forward pass
    std::vector<float> output(nn->layers[nn->num_layers - 1].output_size);
    nn_forward(nn, input, output.data());
    
    // Compute loss
    float loss = 0.0f;
    DenseLayer* last = &nn->layers[nn->num_layers - 1];
    for (int i = 0; i < last->output_size; i++) {
        float diff = output[i] - target[i];
        loss += diff * diff;
    }
    loss /= last->output_size;
    
    // Backward pass
    int output_offset = 0;
    for (int l = 0; l < nn->num_layers; l++) {
        output_offset += nn->layers[l].output_size;
    }
    
    // Output layer delta
    float* delta = nn->layer_deltas + output_offset - last->output_size;
    for (int i = 0; i < last->output_size; i++) {
        delta[i] = (output[i] - target[i]) * 2.0f / last->output_size;
    }
    
    // Backpropagate
    for (int l = nn->num_layers - 1; l >= 0; l--) {
        DenseLayer* layer = &nn->layers[l];
        output_offset -= layer->output_size;
        
        const float* layer_input = (l == 0) ? input : 
            (nn->layer_outputs + output_offset - nn->layers[l - 1].output_size);
        float* layer_delta = nn->layer_deltas + output_offset;
        
        // Compute gradients
        for (int i = 0; i < layer->input_size; i++) {
            for (int j = 0; j < layer->output_size; j++) {
                layer->weight_gradients[i * layer->output_size + j] += 
                    layer_input[i] * layer_delta[j];
            }
        }
        
        for (int j = 0; j < layer->output_size; j++) {
            layer->bias_gradients[j] += layer_delta[j];
        }
        
        // Propagate delta to previous layer
        if (l > 0) {
            DenseLayer* prev = &nn->layers[l - 1];
            float* prev_delta = nn->layer_deltas + output_offset - prev->output_size;
            float* prev_output = nn->layer_outputs + output_offset - prev->output_size;
            
            for (int i = 0; i < prev->output_size; i++) {
                float sum = 0.0f;
                for (int j = 0; j < layer->output_size; j++) {
                    sum += layer->weights[i * layer->output_size + j] * layer_delta[j];
                }
                
                // Apply activation derivative
                switch (prev->activation) {
                    case ACTIVATION_RELU: sum *= relu_deriv(prev_output[i]); break;
                    case ACTIVATION_SIGMOID: sum *= sigmoid_deriv(prev_output[i]); break;
                    case ACTIVATION_TANH: sum *= tanh_deriv(prev_output[i]); break;
                    default: break;
                }
                
                prev_delta[i] = sum;
            }
        }
    }
    
    // Apply gradients
    for (int l = 0; l < nn->num_layers; l++) {
        DenseLayer* layer = &nn->layers[l];
        
        for (int i = 0; i < layer->input_size * layer->output_size; i++) {
            layer->weights[i] -= nn->learning_rate * layer->weight_gradients[i];
            layer->weight_gradients[i] = 0.0f;
        }
        
        for (int j = 0; j < layer->output_size; j++) {
            layer->biases[j] -= nn->learning_rate * layer->bias_gradients[j];
            layer->bias_gradients[j] = 0.0f;
        }
    }
    
    return loss;
}

EXPORT float nn_train_batch(
    NeuralNetwork* nn,
    const float* inputs,
    const float* targets,
    int batch_size
) {
    float total_loss = 0.0f;
    
    int input_size = nn->layers[0].input_size;
    int output_size = nn->layers[nn->num_layers - 1].output_size;
    
    for (int b = 0; b < batch_size; b++) {
        total_loss += nn_backward(nn, 
                                   inputs + b * input_size,
                                   targets + b * output_size);
    }
    
    return total_loss / batch_size;
}

EXPORT void nn_predict(const NeuralNetwork* nn, const float* input, float* output) {
    nn_forward((NeuralNetwork*)nn, input, output);
}

EXPORT int nn_save(const NeuralNetwork* nn, const char* path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return 0;
    
    file.write((char*)&nn->num_layers, sizeof(int));
    file.write((char*)&nn->learning_rate, sizeof(float));
    
    for (int l = 0; l < nn->num_layers; l++) {
        DenseLayer* layer = &nn->layers[l];
        file.write((char*)&layer->input_size, sizeof(int));
        file.write((char*)&layer->output_size, sizeof(int));
        file.write((char*)&layer->activation, sizeof(int));
        
        int num_weights = layer->input_size * layer->output_size;
        file.write((char*)layer->weights, num_weights * sizeof(float));
        file.write((char*)layer->biases, layer->output_size * sizeof(float));
    }
    
    file.close();
    return 1;
}

EXPORT NeuralNetwork* nn_load(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return nullptr;
    
    int num_layers;
    float learning_rate;
    file.read((char*)&num_layers, sizeof(int));
    file.read((char*)&learning_rate, sizeof(float));
    
    std::vector<int> layer_sizes(num_layers + 1);
    std::vector<int> activations(num_layers);
    
    // Placeholder - would need to read layer info
    return nullptr;
}

// ============================================================================
// OPTIMIZER
// ============================================================================

EXPORT Optimizer* create_optimizer(OptimizerType type, int num_params, float learning_rate) {
    Optimizer* opt = new Optimizer();
    opt->type = type;
    opt->learning_rate = learning_rate;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->epsilon = 1e-8f;
    opt->cache_size = num_params;
    opt->step = 0;
    
    opt->m_cache = new float[num_params];
    opt->v_cache = new float[num_params];
    memset(opt->m_cache, 0, num_params * sizeof(float));
    memset(opt->v_cache, 0, num_params * sizeof(float));
    
    return opt;
}

EXPORT void free_optimizer(Optimizer* opt) {
    if (opt) {
        delete[] opt->m_cache;
        delete[] opt->v_cache;
        delete opt;
    }
}

EXPORT void optimizer_step(
    Optimizer* opt,
    float* params,
    const float* gradients,
    int num_params
) {
    opt->step++;
    
    switch (opt->type) {
        case OPTIMIZER_SGD:
            for (int i = 0; i < num_params; i++) {
                params[i] -= opt->learning_rate * gradients[i];
            }
            break;
            
        case OPTIMIZER_MOMENTUM:
            for (int i = 0; i < num_params; i++) {
                opt->m_cache[i] = opt->beta1 * opt->m_cache[i] + (1 - opt->beta1) * gradients[i];
                params[i] -= opt->learning_rate * opt->m_cache[i];
            }
            break;
            
        case OPTIMIZER_ADAM:
            for (int i = 0; i < num_params; i++) {
                opt->m_cache[i] = opt->beta1 * opt->m_cache[i] + (1 - opt->beta1) * gradients[i];
                opt->v_cache[i] = opt->beta2 * opt->v_cache[i] + (1 - opt->beta2) * gradients[i] * gradients[i];
                
                float m_hat = opt->m_cache[i] / (1 - std::pow(opt->beta1, opt->step));
                float v_hat = opt->v_cache[i] / (1 - std::pow(opt->beta2, opt->step));
                
                params[i] -= opt->learning_rate * m_hat / (std::sqrt(v_hat) + opt->epsilon);
            }
            break;
            
        case OPTIMIZER_RMSPROP:
            for (int i = 0; i < num_params; i++) {
                opt->v_cache[i] = opt->beta2 * opt->v_cache[i] + (1 - opt->beta2) * gradients[i] * gradients[i];
                params[i] -= opt->learning_rate * gradients[i] / (std::sqrt(opt->v_cache[i]) + opt->epsilon);
            }
            break;
            
        default:
            break;
    }
}

EXPORT void optimizer_reset(Optimizer* opt) {
    opt->step = 0;
    memset(opt->m_cache, 0, opt->cache_size * sizeof(float));
    memset(opt->v_cache, 0, opt->cache_size * sizeof(float));
}

EXPORT void gradient_descent(
    const float* inputs,
    const float* targets,
    float* weights,
    int num_samples,
    int num_features,
    float learning_rate,
    int iterations
) {
    for (int iter = 0; iter < iterations; iter++) {
        std::vector<float> gradients(num_features, 0.0f);
        
        for (int s = 0; s < num_samples; s++) {
            float pred = 0.0f;
            for (int f = 0; f < num_features; f++) {
                pred += inputs[s * num_features + f] * weights[f];
            }
            
            float error = pred - targets[s];
            
            for (int f = 0; f < num_features; f++) {
                gradients[f] += error * inputs[s * num_features + f];
            }
        }
        
        for (int f = 0; f < num_features; f++) {
            weights[f] -= learning_rate * gradients[f] / num_samples;
        }
    }
}

// ============================================================================
// EVOLUTIONARY ALGORITHMS
// ============================================================================

EXPORT Population* create_population(int size, int gene_length, float gene_min, float gene_max) {
    Population* pop = new Population();
    pop->size = size;
    pop->gene_length = gene_length;
    pop->generation = 0;
    pop->best_fitness = -1e9;
    
    pop->individuals = new Individual[size];
    pop->best_individual = new Individual();
    pop->best_individual->genes = new float[gene_length];
    pop->best_individual->gene_length = gene_length;
    
    std::uniform_real_distribution<float> dist(gene_min, gene_max);
    
    for (int i = 0; i < size; i++) {
        pop->individuals[i].genes = new float[gene_length];
        pop->individuals[i].gene_length = gene_length;
        pop->individuals[i].fitness = 0.0;
        pop->individuals[i].age = 0;
        
        for (int g = 0; g < gene_length; g++) {
            pop->individuals[i].genes[g] = dist(g_gen);
        }
    }
    
    return pop;
}

EXPORT void free_population(Population* pop) {
    if (pop) {
        for (int i = 0; i < pop->size; i++) {
            delete[] pop->individuals[i].genes;
        }
        delete[] pop->individuals;
        delete[] pop->best_individual->genes;
        delete pop->best_individual;
        delete pop;
    }
}

EXPORT void evolve_population(
    Population* pop,
    double (*fitness_func)(const float*, int),
    double mutation_rate,
    double crossover_rate,
    int elitism_count
) {
    // Evaluate fitness
    for (int i = 0; i < pop->size; i++) {
        pop->individuals[i].fitness = fitness_func(pop->individuals[i].genes, pop->gene_length);
        
        if (pop->individuals[i].fitness > pop->best_fitness) {
            pop->best_fitness = pop->individuals[i].fitness;
            memcpy(pop->best_individual->genes, pop->individuals[i].genes, 
                   pop->gene_length * sizeof(float));
            pop->best_individual->fitness = pop->individuals[i].fitness;
        }
    }
    
    // Sort by fitness
    std::sort(pop->individuals, pop->individuals + pop->size,
              [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });
    
    // Create new population
    std::vector<Individual> new_pop(pop->size);
    
    // Elitism
    for (int i = 0; i < elitism_count; i++) {
        new_pop[i].genes = new float[pop->gene_length];
        new_pop[i].gene_length = pop->gene_length;
        memcpy(new_pop[i].genes, pop->individuals[i].genes, pop->gene_length * sizeof(float));
        new_pop[i].fitness = pop->individuals[i].fitness;
        new_pop[i].age = pop->individuals[i].age + 1;
    }
    
    // Crossover and mutation for rest
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    
    for (int i = elitism_count; i < pop->size; i++) {
        int p1 = tournament_select(pop, 3);
        int p2 = tournament_select(pop, 3);
        
        new_pop[i].genes = new float[pop->gene_length];
        new_pop[i].gene_length = pop->gene_length;
        new_pop[i].age = 0;
        
        // Crossover
        if (prob(g_gen) < crossover_rate) {
            int crossover_point = g_gen() % pop->gene_length;
            for (int g = 0; g < pop->gene_length; g++) {
                new_pop[i].genes[g] = (g < crossover_point) ? 
                    pop->individuals[p1].genes[g] : pop->individuals[p2].genes[g];
            }
        } else {
            memcpy(new_pop[i].genes, pop->individuals[p1].genes, pop->gene_length * sizeof(float));
        }
        
        // Mutation
        mutate(&new_pop[i], mutation_rate, 0.1f);
    }
    
    // Replace old population
    for (int i = 0; i < pop->size; i++) {
        delete[] pop->individuals[i].genes;
        pop->individuals[i] = new_pop[i];
    }
    
    pop->generation++;
}

EXPORT int tournament_select(const Population* pop, int tournament_size) {
    int best = g_gen() % pop->size;
    
    for (int i = 1; i < tournament_size; i++) {
        int candidate = g_gen() % pop->size;
        if (pop->individuals[candidate].fitness > pop->individuals[best].fitness) {
            best = candidate;
        }
    }
    
    return best;
}

EXPORT void crossover(
    const Individual* parent1,
    const Individual* parent2,
    Individual* child1,
    Individual* child2
) {
    int crossover_point = g_gen() % parent1->gene_length;
    
    for (int g = 0; g < parent1->gene_length; g++) {
        if (g < crossover_point) {
            child1->genes[g] = parent1->genes[g];
            child2->genes[g] = parent2->genes[g];
        } else {
            child1->genes[g] = parent2->genes[g];
            child2->genes[g] = parent1->genes[g];
        }
    }
}

EXPORT void mutate(Individual* individual, double mutation_rate, float mutation_strength) {
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::normal_distribution<float> normal(0.0f, mutation_strength);
    
    for (int g = 0; g < individual->gene_length; g++) {
        if (prob(g_gen) < mutation_rate) {
            individual->genes[g] += normal(g_gen);
        }
    }
}

EXPORT void get_best_individual(const Population* pop, float* genes_out, double* fitness_out) {
    memcpy(genes_out, pop->best_individual->genes, pop->gene_length * sizeof(float));
    *fitness_out = pop->best_fitness;
}

// ============================================================================
// CMA-ES
// ============================================================================

EXPORT CMAES* create_cmaes(int dimension, const float* initial_mean, float initial_sigma) {
    CMAES* cma = new CMAES();
    cma->dimension = dimension;
    cma->lambda = 4 + (int)(3 * std::log(dimension));
    cma->generation = 0;
    
    cma->mean = new float[dimension];
    cma->sigma = new float[dimension];
    memcpy(cma->mean, initial_mean, dimension * sizeof(float));
    for (int i = 0; i < dimension; i++) cma->sigma[i] = initial_sigma;
    
    cma->C = new float*[dimension];
    cma->B = new float*[dimension];
    cma->D = new float[dimension];
    
    for (int i = 0; i < dimension; i++) {
        cma->C[i] = new float[dimension];
        cma->B[i] = new float[dimension];
        
        for (int j = 0; j < dimension; j++) {
            cma->C[i][j] = (i == j) ? 1.0f : 0.0f;
            cma->B[i][j] = (i == j) ? 1.0f : 0.0f;
        }
        cma->D[i] = 1.0f;
    }
    
    cma->population = new float*[cma->lambda];
    cma->fitness = new float[cma->lambda];
    
    for (int i = 0; i < cma->lambda; i++) {
        cma->population[i] = new float[dimension];
    }
    
    return cma;
}

EXPORT void free_cmaes(CMAES* cma) {
    if (cma) {
        delete[] cma->mean;
        delete[] cma->sigma;
        delete[] cma->D;
        delete[] cma->fitness;
        
        for (int i = 0; i < cma->dimension; i++) {
            delete[] cma->C[i];
            delete[] cma->B[i];
        }
        delete[] cma->C;
        delete[] cma->B;
        
        for (int i = 0; i < cma->lambda; i++) {
            delete[] cma->population[i];
        }
        delete[] cma->population;
        
        delete cma;
    }
}

EXPORT void cmaes_sample(CMAES* cma) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    
    for (int l = 0; l < cma->lambda; l++) {
        for (int d = 0; d < cma->dimension; d++) {
            cma->population[l][d] = cma->mean[d] + cma->sigma[d] * cma->D[d] * normal(g_gen);
        }
    }
}

EXPORT void cmaes_update(CMAES* cma, const float* fitnesses) {
    memcpy(cma->fitness, fitnesses, cma->lambda * sizeof(float));
    
    // Sort by fitness
    std::vector<int> indices(cma->lambda);
    for (int i = 0; i < cma->lambda; i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return cma->fitness[a] > cma->fitness[b]; });
    
    // Update mean using best half
    int mu = cma->lambda / 2;
    for (int d = 0; d < cma->dimension; d++) {
        float new_mean = 0.0f;
        for (int i = 0; i < mu; i++) {
            new_mean += cma->population[indices[i]][d];
        }
        cma->mean[d] = new_mean / mu;
    }
    
    cma->generation++;
}

EXPORT void cmaes_get_best(const CMAES* cma, float* best_out, float* fitness_out) {
    int best_idx = 0;
    float best_fit = cma->fitness[0];
    
    for (int i = 1; i < cma->lambda; i++) {
        if (cma->fitness[i] > best_fit) {
            best_fit = cma->fitness[i];
            best_idx = i;
        }
    }
    
    memcpy(best_out, cma->population[best_idx], cma->dimension * sizeof(float));
    *fitness_out = best_fit;
}

// ============================================================================
// REINFORCEMENT LEARNING
// ============================================================================

EXPORT QTable* create_qtable(int num_states, int num_actions) {
    QTable* qt = new QTable();
    qt->num_states = num_states;
    qt->num_actions = num_actions;
    qt->table = new float[num_states * num_actions];
    memset(qt->table, 0, num_states * num_actions * sizeof(float));
    return qt;
}

EXPORT void free_qtable(QTable* qt) {
    if (qt) {
        delete[] qt->table;
        delete qt;
    }
}

EXPORT void q_learning_update(
    QTable* qt,
    int state,
    int action,
    float reward,
    int next_state,
    float learning_rate,
    float discount_factor
) {
    // Find max Q for next state
    float max_q = qt->table[next_state * qt->num_actions];
    for (int a = 1; a < qt->num_actions; a++) {
        float q = qt->table[next_state * qt->num_actions + a];
        if (q > max_q) max_q = q;
    }
    
    // Update Q value
    int idx = state * qt->num_actions + action;
    qt->table[idx] += learning_rate * (reward + discount_factor * max_q - qt->table[idx]);
}

EXPORT void sarsa_update(
    QTable* qt,
    int state,
    int action,
    float reward,
    int next_state,
    int next_action,
    float learning_rate,
    float discount_factor
) {
    int idx = state * qt->num_actions + action;
    float next_q = qt->table[next_state * qt->num_actions + next_action];
    qt->table[idx] += learning_rate * (reward + discount_factor * next_q - qt->table[idx]);
}

EXPORT int get_best_action(const QTable* qt, int state) {
    int best_action = 0;
    float best_q = qt->table[state * qt->num_actions];
    
    for (int a = 1; a < qt->num_actions; a++) {
        float q = qt->table[state * qt->num_actions + a];
        if (q > best_q) {
            best_q = q;
            best_action = a;
        }
    }
    
    return best_action;
}

EXPORT int epsilon_greedy_action(const QTable* qt, int state, float epsilon) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    
    if (prob(g_gen) < epsilon) {
        return g_gen() % qt->num_actions;
    } else {
        return get_best_action(qt, state);
    }
}

// ============================================================================
// REPLAY BUFFER
// ============================================================================

EXPORT ReplayBuffer* create_replay_buffer(int capacity, int state_dim) {
    ReplayBuffer* rb = new ReplayBuffer();
    rb->capacity = capacity;
    rb->size = 0;
    rb->position = 0;
    rb->state_dim = state_dim;
    
    rb->buffer = new Experience[capacity];
    for (int i = 0; i < capacity; i++) {
        rb->buffer[i].state = new float[state_dim];
        rb->buffer[i].next_state = new float[state_dim];
    }
    
    return rb;
}

EXPORT void free_replay_buffer(ReplayBuffer* rb) {
    if (rb) {
        for (int i = 0; i < rb->capacity; i++) {
            delete[] rb->buffer[i].state;
            delete[] rb->buffer[i].next_state;
        }
        delete[] rb->buffer;
        delete rb;
    }
}

EXPORT void replay_buffer_add(
    ReplayBuffer* rb,
    const float* state,
    int action,
    float reward,
    const float* next_state,
    int done
) {
    Experience* exp = &rb->buffer[rb->position];
    
    memcpy(exp->state, state, rb->state_dim * sizeof(float));
    exp->action = action;
    exp->reward = reward;
    memcpy(exp->next_state, next_state, rb->state_dim * sizeof(float));
    exp->done = done;
    
    rb->position = (rb->position + 1) % rb->capacity;
    if (rb->size < rb->capacity) rb->size++;
}

EXPORT int replay_buffer_sample(
    const ReplayBuffer* rb,
    int batch_size,
    Experience* batch_out
) {
    if (rb->size < batch_size) return 0;
    
    for (int b = 0; b < batch_size; b++) {
        int idx = g_gen() % rb->size;
        batch_out[b] = rb->buffer[idx];
    }
    
    return batch_size;
}

// ============================================================================
// UTILITY
// ============================================================================

EXPORT const char* get_learning_version() {
    return LEARNING_VERSION;
}

EXPORT double benchmark_learning_performance() {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simple benchmark
    int layer_sizes[] = {64, 128, 64, 10};
    int activations[] = {ACTIVATION_RELU, ACTIVATION_RELU, ACTIVATION_SOFTMAX};
    
    NeuralNetwork* nn = create_neural_network(layer_sizes, activations, 4, 0.001f);
    
    float input[64], output[10], target[10];
    for (int i = 0; i < 64; i++) input[i] = 0.1f;
    for (int i = 0; i < 10; i++) target[i] = 0.1f;
    
    for (int i = 0; i < 100; i++) {
        nn_backward(nn, input, target);
    }
    
    free_neural_network(nn);
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}
