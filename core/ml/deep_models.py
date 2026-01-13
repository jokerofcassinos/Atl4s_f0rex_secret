import numpy as np
import logging
import json
import os
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("DeepModels")

# --- ACTIVATION FUNCTIONS ---
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / (e_x.sum(axis=-1, keepdims=True) + 1e-9)

# --- LAYERS ---
class Layer:
    def forward(self, x):
        pass
    def get_weights(self):
        return []
    def set_weights(self, weights):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size, activation='relu'):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0/input_size)
        self.b = np.zeros(output_size)
        self.activation = activation
        
    def forward(self, x):
        z = np.dot(x, self.W) + self.b
        if self.activation == 'relu': return relu(z)
        if self.activation == 'sigmoid': return sigmoid(z)
        if self.activation == 'tanh': return tanh(z)
        if self.activation == 'softmax': return softmax(z)
        return z

    def get_weights(self):
        return [self.W, self.b]

    def set_weights(self, weights):
        self.W = weights[0]
        self.b = weights[1]

class LSTMCell(Layer):
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.concat_size = input_size + hidden_size
        
        # Weights for Forget, Input, Output, Cell Gates
        # Concatenated for efficiency: [Wf, Wi, Wc, Wo]
        self.W = np.random.randn(self.concat_size, hidden_size * 4) * 0.1
        self.b = np.zeros(hidden_size * 4)
        
    def forward(self, x, h_prev=None, c_prev=None):
        if h_prev is None: h_prev = np.zeros(self.hidden_size)
        if c_prev is None: c_prev = np.zeros(self.hidden_size)
        
        # Concatenate Input and Hidden
        concat = np.concatenate((x, h_prev))
        
        # Compute Gates
        gates = np.dot(concat, self.W) + self.b
        
        # Split Gates
        f = sigmoid(gates[:self.hidden_size])
        i = sigmoid(gates[self.hidden_size:self.hidden_size*2])
        c_tilde = tanh(gates[self.hidden_size*2:self.hidden_size*3])
        o = sigmoid(gates[self.hidden_size*3:])
        
        # Update Cell
        c_next = f * c_prev + i * c_tilde
        h_next = o * tanh(c_next)
        
        return h_next, c_next

    def get_weights(self):
        return [self.W, self.b]
        
    def set_weights(self, weights):
        self.W = weights[0]
        self.b = weights[1]

# --- MODEL CONTAINER ---
class DeepModel:
    def __init__(self, name: str):
        self.name = name
        self.layers: List[Layer] = []
        
    def add(self, layer: Layer):
        self.layers.append(layer)
        
    def predict(self, x):
        out = x
        for layer in self.layers:
            if isinstance(layer, LSTMCell):
                # Simple sequence handling (just last step or accumulation)
                # For basic use, we ignore sequence structure here or assume X is flat?
                # Actually, standard Dense MLP pattern.
                # If LSTM, we need State Management.
                # Simplification: This container is for FEED FORWARD unless specialized.
                pass
            
            if not isinstance(layer, LSTMCell):
                out = layer.forward(out)
        return out

class LSTMModel(DeepModel):
    def __init__(self, name, input_size, hidden_size, output_size):
        super().__init__(name)
        self.lstm = LSTMCell(input_size, hidden_size)
        self.dense = Dense(hidden_size, output_size, activation='softmax' if output_size > 1 else 'tanh')
        self.h = None
        self.c = None
        self.hidden_size = hidden_size
        
    def predict(self, x):
        # x is single timestep vector
        if self.h is None: 
            self.h = np.zeros(self.hidden_size)
            self.c = np.zeros(self.hidden_size)
            
        self.h, self.c = self.lstm.forward(x, self.h, self.c)
        return self.dense.forward(self.h)
        
    def reset(self):
        self.h = None
        self.c = None

# --- MODEL HUB ---
class ModelHub:
    """
    Managing the Neural Zoo.
    """
    def __init__(self):
        self.models: Dict[str, DeepModel] = {}
        
    def create_price_predictor(self):
        # Input: [RSI, MACD, Z-Score, PriceDelta] (4 features)
        model = LSTMModel("PriceOracle", 4, 16, 1) # Output: Next Tick Delta Prediction
        self.models['price_predictor'] = model
        return model
        
    def get_prediction(self, model_name, inputs):
        if model_name in self.models:
            return self.models[model_name].predict(np.array(inputs))
        return 0.0

