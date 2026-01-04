
import math
import random
import logging
from typing import List, Dict, Any, Optional
import numpy as np

# We used the Oracle for simulation
from analysis.monte_carlo_fractal import FractalMonteCarlo

logger = logging.getLogger("MCTS_Planner")

class MCTSNode:
    def __init__(self, state: Dict, parent=None, move=None):
        self.state = state # Market State (Simulated)
        self.parent = parent
        self.move = move # Action taken to reach this node (HOLD, CLOSE, ADD)
        self.children = []
        self.wins = 0.0 # Accumulated Value (EV)
        self.visits = 0
        self.untried_moves = ["HOLD", "CLOSE"] # Simplified Action Space

    def uct_select_child(self, exploration_weight=1.41):
        """Upper Confidence Bound 1 applied to trees."""
         # UCT = avg_win + C * sqrt(ln(N) / n)
        s = sorted(self.children, key=lambda c: c.wins / c.visits + exploration_weight * math.sqrt(math.log(self.visits) / c.visits))
        return s[-1]

    def add_child(self, move, state):
        child = MCTSNode(state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

class MCTSPlanner:
    """
    The Grandmaster.
    Uses Monte Carlo Tree Search to plan optimal trade management.
    """
    def __init__(self):
        self.oracle = FractalMonteCarlo()
        self.simulation_depth = 5 # Look 5 steps (candles) ahead
        self.iterations = 50 # Simulations per decision (HFT constrained)

    def _run_simulation_bridge(self, root_state: Dict, drift: float) -> str:
        """
        Attempts to run MCTS via C++ Engine. Falls back to Python.
        """
        # Try Loading C++ DLL
        try:
            import ctypes
            import os
            
            dll_path = os.path.join("cpp_core", "mcts_core.dll")
            if not os.path.exists(dll_path):
                raise FileNotFoundError("DLL not compiled")
                
            lib = ctypes.CDLL(dll_path)
            
            # Signature: run_mcts_simulation(price, entry, dir, vol, drift, iter, depth)
            lib.run_mcts_simulation.argtypes = [
                ctypes.c_double, ctypes.c_double, ctypes.c_int,
                ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int
            ]
            
            class SimResult(ctypes.Structure):
                _fields_ = [
                    ("best_move_type", ctypes.c_int),
                    ("expected_value", ctypes.c_double),
                    ("visits", ctypes.c_int)
                ]
            lib.run_mcts_simulation.restype = SimResult
            
            # Prepare Args
            price = float(root_state['price'])
            entry = float(root_state['entry'])
            direction = 1 if root_state['side'] == 'BUY' else -1
            vol = float(root_state.get('volatility', 1.0))
            
            # Call C++
            # Boost iterations if C++ is active!
            turbo_iters = self.iterations * 100 
            
            res = lib.run_mcts_simulation(price, entry, direction, vol, drift, turbo_iters, self.simulation_depth)
            
            move_map = {0: "HOLD", 1: "CLOSE", 2: "ADD"}
            best_move = move_map.get(res.best_move_type, "HOLD")
            
            logger.debug(f"MCTS [C++]: {best_move} (Visits: {res.visits}, EV: {res.expected_value:.2f})")
            return best_move
            
        except Exception as e:
            # Fallback to Python
            # logger.warning(f"MCTS C++ Unavailable ({e}). Using Python Fallback.")
            return self._run_simulation_python(root_state, drift)

    def _run_simulation_python(self, root_state: Dict, drift: float) -> str:
        """Original Python Implementation."""
        root_state['projected_drift'] = drift
        root = MCTSNode(state=root_state)
        
        for _ in range(self.iterations):
            node = root
            state = root_state.copy()
            
            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.uct_select_child()
                
            # Expansion
            if node.untried_moves != []:
                m = random.choice(node.untried_moves) 
                state = self._simulate_step(state, m)
                node = node.add_child(m, state)
                
            # Rollout
            depth = 0
            while depth < self.simulation_depth:
                possible_moves = ["HOLD", "CLOSE"]
                m = random.choice(possible_moves)
                state = self._simulate_step(state, m)
                if m == "CLOSE": break
                depth += 1
                
            # Backprop
            reward = self._evaluate_terminal_state(state)
            while node != None:
                node.visits += 1
                node.wins += reward
                node = node.parent
                
        if not root.children: return "HOLD"
        best_node = sorted(root.children, key=lambda c: c.visits)[-1]
        logger.debug(f"MCTS [PY]: {best_node.move} (Visits: {best_node.visits}, EV: {best_node.wins/best_node.visits:.2f})")
        return best_node.move

    def search(self, root_state: Dict, trend_bias: float = 0.0) -> str:
        return self._run_simulation_bridge(root_state, trend_bias)

    def _simulate_step(self, state: Dict, move: str) -> Dict:
        # Helper for Python loop
        new_state = state.copy()
        if move == "CLOSE":
            new_state['active'] = False
            return new_state
        if move == "HOLD":
            drift = new_state.get('projected_drift', 0.0) 
            vol = new_state.get('volatility', 1.0)
            shock = np.random.normal(0, vol)
            new_state['price'] += drift + shock
            entry = new_state['entry']
            if new_state['side'] == 'BUY': new_state['pnl'] = new_state['price'] - entry
            else: new_state['pnl'] = entry - new_state['price']
            new_state['active'] = True
        return new_state

    def _evaluate_terminal_state(self, state: Dict) -> float:
        return state['pnl'] 
