
#ifndef LEARNING_H
#define LEARNING_H

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

#include <cstdint>

extern "C" {
    // ============================================================================
    // LEARNING CORE - AGI C++ Implementation
    // Neural networks, gradient descent, evolutionary algorithms, RL
    // ============================================================================

    // ============================================================================
    // NEURAL NETWORK
    // ============================================================================
    
    enum ActivationType {
        ACTIVATION_RELU = 0,
        ACTIVATION_SIGMOID = 1,
        ACTIVATION_TANH = 2,
        ACTIVATION_SOFTMAX = 3,
        ACTIVATION_LINEAR = 4
    };
    
    struct DenseLayer {
        float* weights;
        float* biases;
        float* weight_gradients;
        float* bias_gradients;
        int input_size;
        int output_size;
        ActivationType activation;
    };
    
    struct NeuralNetwork {
        DenseLayer* layers;
        int num_layers;
        float learning_rate;
        float momentum;
        float* layer_outputs;
        float* layer_deltas;
    };
    
    // Create neural network
    EXPORT NeuralNetwork* create_neural_network(
        const int* layer_sizes,
        const int* activations,
        int num_layers,
        float learning_rate
    );
    
    // Free neural network
    EXPORT void free_neural_network(NeuralNetwork* nn);
    
    // Forward pass
    EXPORT void nn_forward(
        NeuralNetwork* nn,
        const float* input,
        float* output
    );
    
    // Backward pass (backpropagation)
    EXPORT float nn_backward(
        NeuralNetwork* nn,
        const float* input,
        const float* target
    );
    
    // Train on batch
    EXPORT float nn_train_batch(
        NeuralNetwork* nn,
        const float* inputs,
        const float* targets,
        int batch_size
    );
    
    // Predict
    EXPORT void nn_predict(
        const NeuralNetwork* nn,
        const float* input,
        float* output
    );
    
    // Save model
    EXPORT int nn_save(const NeuralNetwork* nn, const char* path);
    
    // Load model
    EXPORT NeuralNetwork* nn_load(const char* path);

    // ============================================================================
    // GRADIENT DESCENT
    // ============================================================================
    
    enum OptimizerType {
        OPTIMIZER_SGD = 0,
        OPTIMIZER_MOMENTUM = 1,
        OPTIMIZER_ADAM = 2,
        OPTIMIZER_RMSPROP = 3,
        OPTIMIZER_ADAGRAD = 4
    };
    
    struct Optimizer {
        OptimizerType type;
        float learning_rate;
        float beta1;
        float beta2;
        float epsilon;
        float* m_cache;  // First moment
        float* v_cache;  // Second moment
        int cache_size;
        int step;
    };
    
    // Create optimizer
    EXPORT Optimizer* create_optimizer(
        OptimizerType type,
        int num_params,
        float learning_rate
    );
    
    // Free optimizer
    EXPORT void free_optimizer(Optimizer* opt);
    
    // Apply gradient update
    EXPORT void optimizer_step(
        Optimizer* opt,
        float* params,
        const float* gradients,
        int num_params
    );
    
    // Reset optimizer state
    EXPORT void optimizer_reset(Optimizer* opt);
    
    // Manual gradient descent
    EXPORT void gradient_descent(
        const float* inputs,
        const float* targets,
        float* weights,
        int num_samples,
        int num_features,
        float learning_rate,
        int iterations
    );

    // ============================================================================
    // EVOLUTIONARY ALGORITHMS
    // ============================================================================
    
    struct Individual {
        float* genes;
        int gene_length;
        double fitness;
        int age;
    };
    
    struct Population {
        Individual* individuals;
        int size;
        int gene_length;
        Individual* best_individual;
        double best_fitness;
        int generation;
    };
    
    // Create population
    EXPORT Population* create_population(
        int size,
        int gene_length,
        float gene_min,
        float gene_max
    );
    
    // Free population
    EXPORT void free_population(Population* pop);
    
    // Evolve population
    EXPORT void evolve_population(
        Population* pop,
        double (*fitness_func)(const float*, int),
        double mutation_rate,
        double crossover_rate,
        int elitism_count
    );
    
    // Tournament selection
    EXPORT int tournament_select(
        const Population* pop,
        int tournament_size
    );
    
    // Crossover (single point)
    EXPORT void crossover(
        const Individual* parent1,
        const Individual* parent2,
        Individual* child1,
        Individual* child2
    );
    
    // Mutation
    EXPORT void mutate(
        Individual* individual,
        double mutation_rate,
        float mutation_strength
    );
    
    // Get best individual
    EXPORT void get_best_individual(
        const Population* pop,
        float* genes_out,
        double* fitness_out
    );

    // ============================================================================
    // CMA-ES (Covariance Matrix Adaptation)
    // ============================================================================
    
    struct CMAES {
        float* mean;
        float* sigma;
        float** C;          // Covariance matrix
        float** B;          // Eigenvectors
        float* D;           // Eigenvalues
        float** population;
        float* fitness;
        int dimension;
        int lambda;         // Population size
        int generation;
    };
    
    // Create CMA-ES
    EXPORT CMAES* create_cmaes(
        int dimension,
        const float* initial_mean,
        float initial_sigma
    );
    
    // Free CMA-ES
    EXPORT void free_cmaes(CMAES* cma);
    
    // Sample population
    EXPORT void cmaes_sample(CMAES* cma);
    
    // Update CMA-ES
    EXPORT void cmaes_update(
        CMAES* cma,
        const float* fitnesses
    );
    
    // Get CMA-ES best
    EXPORT void cmaes_get_best(
        const CMAES* cma,
        float* best_out,
        float* fitness_out
    );

    // ============================================================================
    // REINFORCEMENT LEARNING
    // ============================================================================
    
    struct QTable {
        float* table;
        int num_states;
        int num_actions;
    };
    
    // Create Q-table
    EXPORT QTable* create_qtable(int num_states, int num_actions);
    
    // Free Q-table
    EXPORT void free_qtable(QTable* qt);
    
    // Q-learning update
    EXPORT void q_learning_update(
        QTable* qt,
        int state,
        int action,
        float reward,
        int next_state,
        float learning_rate,
        float discount_factor
    );
    
    // SARSA update
    EXPORT void sarsa_update(
        QTable* qt,
        int state,
        int action,
        float reward,
        int next_state,
        int next_action,
        float learning_rate,
        float discount_factor
    );
    
    // Get best action
    EXPORT int get_best_action(const QTable* qt, int state);
    
    // Epsilon-greedy action
    EXPORT int epsilon_greedy_action(
        const QTable* qt,
        int state,
        float epsilon
    );
    
    // DQN Experience
    struct Experience {
        float* state;
        int action;
        float reward;
        float* next_state;
        int done;
    };
    
    struct ReplayBuffer {
        Experience* buffer;
        int capacity;
        int size;
        int position;
        int state_dim;
    };
    
    // Create replay buffer
    EXPORT ReplayBuffer* create_replay_buffer(int capacity, int state_dim);
    
    // Free replay buffer
    EXPORT void free_replay_buffer(ReplayBuffer* rb);
    
    // Add experience
    EXPORT void replay_buffer_add(
        ReplayBuffer* rb,
        const float* state,
        int action,
        float reward,
        const float* next_state,
        int done
    );
    
    // Sample batch
    EXPORT int replay_buffer_sample(
        const ReplayBuffer* rb,
        int batch_size,
        Experience* batch_out
    );

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================
    
    // Set random seed
    EXPORT void set_learning_seed(unsigned int seed);
    
    // Get version
    EXPORT const char* get_learning_version();
    
    // Benchmark
    EXPORT double benchmark_learning_performance();
}

#endif
