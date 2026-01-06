"""
AGI Ultra: Meta-Learning Engine - Learning to Learn

Implements sophisticated meta-learning capabilities:
- Transfer learning between market regimes
- Few-shot adaptation for new conditions
- Curriculum learning for gradual skill building
- Self-improvement metric tracking
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import os
import pickle

logger = logging.getLogger("MetaLearningEngine")


@dataclass
class LearningEpisode:
    """A single learning experience."""
    episode_id: str
    task_type: str  # regime_adaptation, pattern_recognition, risk_assessment
    context: Dict[str, Any]
    
    # Performance metrics
    start_time: float
    end_time: Optional[float] = None
    steps_taken: int = 0
    final_score: float = 0.0
    improvement_rate: float = 0.0
    
    # Learning trajectory
    score_history: List[float] = field(default_factory=list)
    gradients: List[np.ndarray] = field(default_factory=list)


@dataclass
class TransferTask:
    """A transfer learning task."""
    source_regime: str
    target_regime: str
    transfer_weights: np.ndarray
    adaptation_score: float = 0.0
    samples_used: int = 0


@dataclass
class CurriculumStage:
    """A stage in curriculum learning."""
    stage_id: int
    name: str
    difficulty: float  # 0.0 to 1.0
    requirements: Dict[str, float]  # metric -> threshold
    completed: bool = False
    completion_time: Optional[float] = None


class MetaLearningEngine:
    """
    AGI Ultra: Meta-Learning Engine.
    
    Enables the system to "learn how to learn" through:
    - Learning rate adaptation based on task similarity
    - Transfer learning between market regimes
    - Few-shot adaptation for new market conditions
    - Curriculum learning for progressive skill development
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        meta_lr: float = 0.01,
        adaptation_steps: int = 5,
        persistence_dir: str = "brain/meta_learning"
    ):
        self.feature_dim = feature_dim
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        self.persistence_dir = persistence_dir
        
        # Meta-parameters (parameters of the learning algorithm itself)
        self.meta_weights = np.random.randn(feature_dim, feature_dim) * 0.01
        self.meta_bias = np.zeros(feature_dim)
        
        # Task embeddings for similarity computation
        self.task_embeddings: Dict[str, np.ndarray] = {}
        
        # Learning history
        self.episodes: List[LearningEpisode] = []
        self.max_episodes = 10000
        
        # Transfer learning state
        self.transfer_tasks: List[TransferTask] = []
        self.regime_adapters: Dict[str, np.ndarray] = {}
        
        # Curriculum learning state
        self.curriculum: List[CurriculumStage] = self._init_curriculum()
        self.current_stage: int = 0
        
        # Performance tracking
        self.metrics = {
            'overall_learning_rate': 0.0,
            'transfer_efficiency': 0.0,
            'adaptation_speed': 0.0,
            'curriculum_progress': 0.0,
            'meta_loss_history': deque(maxlen=1000)
        }
        
        os.makedirs(persistence_dir, exist_ok=True)
        logger.info(f"MetaLearningEngine initialized: feature_dim={feature_dim}, meta_lr={meta_lr}")
    
    def _init_curriculum(self) -> List[CurriculumStage]:
        """Initialize curriculum learning stages."""
        return [
            CurriculumStage(
                stage_id=0,
                name="basic_pattern_recognition",
                difficulty=0.1,
                requirements={'pattern_accuracy': 0.6}
            ),
            CurriculumStage(
                stage_id=1,
                name="trend_identification",
                difficulty=0.2,
                requirements={'trend_accuracy': 0.65, 'pattern_accuracy': 0.65}
            ),
            CurriculumStage(
                stage_id=2,
                name="regime_classification",
                difficulty=0.4,
                requirements={'regime_accuracy': 0.7, 'trend_accuracy': 0.7}
            ),
            CurriculumStage(
                stage_id=3,
                name="risk_assessment",
                difficulty=0.5,
                requirements={'risk_accuracy': 0.7, 'regime_accuracy': 0.7}
            ),
            CurriculumStage(
                stage_id=4,
                name="entry_timing",
                difficulty=0.6,
                requirements={'entry_accuracy': 0.65, 'risk_accuracy': 0.7}
            ),
            CurriculumStage(
                stage_id=5,
                name="position_sizing",
                difficulty=0.7,
                requirements={'sizing_accuracy': 0.7, 'entry_accuracy': 0.65}
            ),
            CurriculumStage(
                stage_id=6,
                name="exit_optimization",
                difficulty=0.8,
                requirements={'exit_accuracy': 0.7, 'sizing_accuracy': 0.7}
            ),
            CurriculumStage(
                stage_id=7,
                name="multi_timeframe_synthesis",
                difficulty=0.9,
                requirements={'mtf_accuracy': 0.7, 'exit_accuracy': 0.7}
            ),
            CurriculumStage(
                stage_id=8,
                name="adaptive_strategy_selection",
                difficulty=0.95,
                requirements={'strategy_accuracy': 0.75, 'mtf_accuracy': 0.7}
            ),
            CurriculumStage(
                stage_id=9,
                name="meta_optimization",
                difficulty=1.0,
                requirements={'overall_profit': 0.6, 'sharpe_ratio': 1.5}
            )
        ]
    
    # -------------------------------------------------------------------------
    # TRANSFER LEARNING
    # -------------------------------------------------------------------------
    def prepare_transfer(
        self,
        source_regime: str,
        source_data: np.ndarray,
        source_labels: np.ndarray
    ) -> np.ndarray:
        """
        Prepare transfer weights from source regime.
        
        Args:
            source_regime: Name of source market regime
            source_data: Training data from source regime
            source_labels: Labels from source regime
            
        Returns:
            Transfer weights to be used for target adaptation
        """
        # Compute regime-specific adapter
        if len(source_data) == 0:
            return self.meta_weights.copy()
        
        # Simple linear adaptation: W_regime = W_meta + alpha * (data correlation)
        data_mean = np.mean(source_data, axis=0)
        data_std = np.std(source_data, axis=0) + 1e-8
        
        # Normalize
        normalized = (source_data - data_mean) / data_std
        
        # Compute correlation-based weights
        if len(normalized) > 1:
            correlation = np.corrcoef(normalized.T)
            correlation = np.nan_to_num(correlation, 0)
        else:
            correlation = np.eye(min(self.feature_dim, normalized.shape[1]))
        
        # Resize if needed
        if correlation.shape[0] != self.feature_dim:
            padded = np.eye(self.feature_dim)
            min_dim = min(correlation.shape[0], self.feature_dim)
            padded[:min_dim, :min_dim] = correlation[:min_dim, :min_dim]
            correlation = padded
        
        adapter = self.meta_weights + 0.1 * correlation
        self.regime_adapters[source_regime] = adapter
        
        logger.debug(f"Prepared transfer from {source_regime}: adapter shape={adapter.shape}")
        return adapter
    
    def adapt_to_target(
        self,
        target_regime: str,
        target_data: np.ndarray,
        target_labels: np.ndarray,
        source_regime: Optional[str] = None,
        num_steps: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Adapt to target regime using transfer learning.
        
        Args:
            target_regime: Target market regime
            target_data: Few-shot data from target
            target_labels: Few-shot labels from target
            source_regime: Source regime to transfer from (optional)
            num_steps: Number of adaptation steps
            
        Returns:
            Tuple of (adapted_weights, adaptation_score)
        """
        num_steps = num_steps or self.adaptation_steps
        
        # Get starting weights
        if source_regime and source_regime in self.regime_adapters:
            weights = self.regime_adapters[source_regime].copy()
        else:
            weights = self.meta_weights.copy()
        
        if len(target_data) == 0:
            return weights, 0.0
        
        # Few-shot adaptation using gradient descent
        best_score = 0.0
        
        for step in range(num_steps):
            # Forward pass
            predictions = self._forward(target_data, weights)
            
            # Compute loss
            loss = np.mean((predictions - target_labels) ** 2)
            
            # Compute gradient (simple gradient for MSE)
            error = predictions - target_labels
            if len(target_data.shape) > 1:
                gradient = np.mean(np.outer(error, target_data.mean(axis=0)), axis=0)
            else:
                gradient = error * target_data.mean()
            
            # Ensure gradient shape matches weights
            if gradient.shape != weights.shape:
                gradient = np.resize(gradient, weights.shape)
            
            # Update weights
            weights = weights - self.meta_lr * gradient
            
            # Track score
            score = 1.0 / (1.0 + loss)
            best_score = max(best_score, score)
        
        # Save adaptation
        self.regime_adapters[target_regime] = weights
        
        self.transfer_tasks.append(TransferTask(
            source_regime=source_regime or "meta",
            target_regime=target_regime,
            transfer_weights=weights,
            adaptation_score=best_score,
            samples_used=len(target_data)
        ))
        
        logger.debug(f"Adapted to {target_regime}: score={best_score:.3f}, steps={num_steps}")
        return weights, best_score
    
    def _forward(self, data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Forward pass through adapted network."""
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        # Resize data if needed
        if data.shape[1] != weights.shape[1]:
            padded = np.zeros((data.shape[0], weights.shape[1]))
            min_dim = min(data.shape[1], weights.shape[1])
            padded[:, :min_dim] = data[:, :min_dim]
            data = padded
        
        # Linear transformation + nonlinearity
        output = np.tanh(data @ weights.T + self.meta_bias)
        
        return output.mean(axis=1) if len(output.shape) > 1 else output
    
    # -------------------------------------------------------------------------
    # FEW-SHOT ADAPTATION
    # -------------------------------------------------------------------------
    def few_shot_adapt(
        self,
        context: Dict[str, Any],
        support_set: List[Tuple[Dict, float]],  # (input, label) pairs
        query_set: Optional[List[Dict]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform few-shot adaptation using support examples.
        
        Args:
            context: Current market context
            support_set: Few labeled examples for adaptation
            query_set: Optional queries to predict
            
        Returns:
            Tuple of (predictions, adaptation_info)
        """
        if len(support_set) == 0:
            return np.array([]), {'error': 'empty support set'}
        
        # Encode support set
        support_data = []
        support_labels = []
        
        for input_dict, label in support_set:
            encoding = self._encode_context(input_dict)
            support_data.append(encoding)
            support_labels.append(label)
        
        support_data = np.array(support_data)
        support_labels = np.array(support_labels)
        
        # Determine regime from context
        regime = context.get('regime', 'unknown')
        
        # Adapt weights
        adapted_weights, score = self.adapt_to_target(
            target_regime=f"fewshot_{regime}",
            target_data=support_data,
            target_labels=support_labels
        )
        
        # Predict on query set if provided
        predictions = np.array([])
        if query_set:
            query_data = np.array([self._encode_context(q) for q in query_set])
            predictions = self._forward(query_data, adapted_weights)
        
        return predictions, {
            'adaptation_score': score,
            'support_size': len(support_set),
            'regime': regime
        }
    
    def _encode_context(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode context dictionary into feature vector."""
        encoding = np.zeros(self.feature_dim)
        
        idx = 0
        for key, value in context.items():
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                if idx < self.feature_dim:
                    encoding[idx] = np.tanh(value * 0.1)
                    idx += 1
        
        return encoding
    
    # -------------------------------------------------------------------------
    # CURRICULUM LEARNING
    # -------------------------------------------------------------------------
    def get_current_curriculum_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        if self.current_stage < len(self.curriculum):
            return self.curriculum[self.current_stage]
        return self.curriculum[-1]
    
    def update_curriculum_progress(self, metrics: Dict[str, float]) -> bool:
        """
        Update curriculum progress based on current metrics.
        
        Args:
            metrics: Current performance metrics
            
        Returns:
            True if stage was completed and advanced
        """
        if self.current_stage >= len(self.curriculum):
            return False
        
        stage = self.curriculum[self.current_stage]
        
        # Check all requirements
        all_met = True
        for metric_name, threshold in stage.requirements.items():
            current_value = metrics.get(metric_name, 0.0)
            if current_value < threshold:
                all_met = False
                break
        
        if all_met and not stage.completed:
            stage.completed = True
            stage.completion_time = time.time()
            self.current_stage += 1
            
            self.metrics['curriculum_progress'] = self.current_stage / len(self.curriculum)
            
            logger.info(f"Curriculum stage completed: {stage.name} -> advancing to stage {self.current_stage}")
            return True
        
        return False
    
    def get_curriculum_task_difficulty(self) -> float:
        """Get current task difficulty based on curriculum."""
        return self.get_current_curriculum_stage().difficulty
    
    # -------------------------------------------------------------------------
    # META-LEARNING OPTIMIZATION
    # -------------------------------------------------------------------------
    def meta_update(
        self,
        episodes: List[LearningEpisode],
        outer_lr: Optional[float] = None
    ):
        """
        Perform meta-update across multiple learning episodes (MAML-style).
        
        Args:
            episodes: List of recent learning episodes
            outer_lr: Outer loop learning rate
        """
        if len(episodes) == 0:
            return
        
        outer_lr = outer_lr or self.meta_lr * 0.1
        
        # Compute meta-gradient from episode gradients
        meta_gradient = np.zeros_like(self.meta_weights)
        
        for episode in episodes:
            if episode.gradients:
                # Average gradients from this episode
                avg_grad = np.mean(episode.gradients, axis=0)
                
                # Weight by improvement rate
                weight = max(0.1, episode.improvement_rate)
                
                # Resize if needed
                if avg_grad.shape != meta_gradient.shape:
                    avg_grad = np.resize(avg_grad, meta_gradient.shape)
                
                meta_gradient += weight * avg_grad
        
        # Normalize
        meta_gradient /= len(episodes)
        
        # Apply meta-update
        self.meta_weights = self.meta_weights - outer_lr * meta_gradient
        
        # Track meta-loss
        meta_loss = np.linalg.norm(meta_gradient)
        self.metrics['meta_loss_history'].append(meta_loss)
        
        # Update learning rate metric
        if len(self.metrics['meta_loss_history']) > 10:
            recent_losses = list(self.metrics['meta_loss_history'])[-10:]
            self.metrics['overall_learning_rate'] = 1.0 / (1.0 + np.mean(recent_losses))
        
        logger.debug(f"Meta-update: gradient_norm={meta_loss:.4f}")
    
    def compute_task_similarity(self, task1_context: Dict, task2_context: Dict) -> float:
        """Compute similarity between two tasks for transfer learning."""
        enc1 = self._encode_context(task1_context)
        enc2 = self._encode_context(task2_context)
        
        norm1 = np.linalg.norm(enc1)
        norm2 = np.linalg.norm(enc2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(enc1, enc2) / (norm1 * norm2))
    
    # -------------------------------------------------------------------------
    # SELF-IMPROVEMENT METRICS
    # -------------------------------------------------------------------------
    def compute_improvement_rate(self, window: int = 100) -> float:
        """Compute recent improvement rate."""
        if len(self.episodes) < 2:
            return 0.0
        
        recent = self.episodes[-window:]
        
        if len(recent) < 2:
            return 0.0
        
        scores = [ep.final_score for ep in recent]
        
        # Linear regression slope
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        return float(slope)
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        stats = {
            'total_episodes': len(self.episodes),
            'transfer_tasks': len(self.transfer_tasks),
            'current_curriculum_stage': self.current_stage,
            'curriculum_progress': self.metrics['curriculum_progress'],
            'overall_learning_rate': self.metrics['overall_learning_rate'],
            'adaptation_speed': self.metrics['adaptation_speed'],
            'regime_adapters': list(self.regime_adapters.keys())
        }
        
        if self.episodes:
            recent = self.episodes[-100:]
            stats['recent_avg_score'] = np.mean([ep.final_score for ep in recent])
            stats['improvement_rate'] = self.compute_improvement_rate()
        
        return stats
    
    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------
    def save(self) -> bool:
        """Save meta-learning state."""
        try:
            filepath = os.path.join(self.persistence_dir, "meta_state.pkl")
            
            state = {
                'meta_weights': self.meta_weights,
                'meta_bias': self.meta_bias,
                'regime_adapters': self.regime_adapters,
                'curriculum': [(s.stage_id, s.completed, s.completion_time) for s in self.curriculum],
                'current_stage': self.current_stage,
                'metrics': dict(self.metrics)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info("MetaLearningEngine state saved")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save meta-learning state: {e}")
            return False
    
    def load(self) -> bool:
        """Load meta-learning state."""
        try:
            filepath = os.path.join(self.persistence_dir, "meta_state.pkl")
            
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.meta_weights = state['meta_weights']
            self.meta_bias = state['meta_bias']
            self.regime_adapters = state['regime_adapters']
            self.current_stage = state['current_stage']
            
            # Restore curriculum state
            for stage_id, completed, completion_time in state.get('curriculum', []):
                if stage_id < len(self.curriculum):
                    self.curriculum[stage_id].completed = completed
                    self.curriculum[stage_id].completion_time = completion_time
            
            logger.info("MetaLearningEngine state loaded")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load meta-learning state: {e}")
            return False
