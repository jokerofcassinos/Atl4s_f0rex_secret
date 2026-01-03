
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

    def search(self, root_state: Dict, trend_bias: float = 0.0) -> str:
        """
        Run MCTS to find the best immediate action.
        root_state: {'price': 100, 'entry': 98, 'pnl': 2.0}
        trend_bias: Drift to apply to simulation (from Swarm Consensus)
        """
        root_state['projected_drift'] = trend_bias # Inject Swarm Opinion
        root = MCTSNode(state=root_state)
        
        # Iteration Budget
        for _ in range(self.iterations):
            node = root
            state = root_state.copy()
            
            # 1. Selection
            while node.untried_moves == [] and node.children != []:
                node = node.uct_select_child()
                # Update implicit state if we were tracking full Simulation path (omitted for speed)
                
            # 2. Expansion
            if node.untried_moves != []:
                m = random.choice(node.untried_moves) 
                # Simulate Next State based on Move
                state = self._simulate_step(state, m)
                node = node.add_child(m, state)
                
            # 3. Simulation (Rollout)
            # Rollout until depth
            depth = 0
            while depth < self.simulation_depth:
                # Random Policy or Heuristic
                possible_moves = ["HOLD", "CLOSE"]
                m = random.choice(possible_moves)
                state = self._simulate_step(state, m)
                
                if m == "CLOSE": break # Terminal state
                depth += 1
                
            # 4. Backpropagation
            # Calculate Reward (PnL / Risk)
            reward = self._evaluate_terminal_state(state)
            while node != None:
                node.visits += 1
                node.wins += reward
                node = node.parent
                
        # Best Move
        if not root.children: return "HOLD" # Default
        
        # Select child with highest visit count (Robust choice)
        best_node = sorted(root.children, key=lambda c: c.visits)[-1]
        logger.debug(f"MCTS Plan: {best_node.move} (Visits: {best_node.visits}, EV: {best_node.wins/best_node.visits:.2f})")
        
        return best_node.move

    def _simulate_step(self, state: Dict, move: str) -> Dict:
        """
        Transitions the state given a move.
        Crucial: Uses Oracle to get 'Expected' Price.
        """
        new_state = state.copy()
        
        if move == "CLOSE":
            # PnL locked at current price
            new_state['active'] = False
            return new_state
            
        if move == "HOLD":
            # Project Price 1 Step Ahead using Oracle
            # (Simplified for speed: Assume generic step from Oracle projection)
            # In full version: call self.oracle.generate_projection() here? Too slow?
            # Speed Hack: Use a pre-calculated 'drift' from the root context.
            
            current_price = new_state['price']
            
            # Simple Random Walk for Rollout if Oracle is too heavy
            # Or use 'projected_drift' stored in state
            drift = new_state.get('projected_drift', 0.0) 
            vol = new_state.get('volatility', 1.0)
            
            shock = np.random.normal(0, vol)
            new_price = current_price + drift + shock
            
            new_state['price'] = new_price
            new_state['pnl'] = (new_price - new_state['entry']) if new_state['side'] == 'BUY' else (new_state['entry'] - new_price)
            new_state['active'] = True
            
        return new_state

    def _evaluate_terminal_state(self, state: Dict) -> float:
        """Returns Utility (Reward)."""
        # Utility = Normalized PnL
        pnl = state['pnl']
        # Normalize? e.g. target is 100 pts.
        return pnl 
