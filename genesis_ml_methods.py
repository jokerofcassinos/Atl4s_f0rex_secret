    def _load_optimizations(self) -> Dict:
        """
        Load ML-optimized parameters
        
        Returns dict of optimized parameter values
        """
        try:
            suggestions = self.ml_optimizer.analyze_optimal_parameters(days=30)
            
            optimized = {}
            for s in suggestions:
                if s.confidence >= 75 and s.expected_improvement >= 5:
                    optimized[s.parameter_name] = s.suggested_value
                    logger.info(f"  📊 Optimization: {s.parameter_name} = {s.suggested_value} (+{s.expected_improvement:.1f}% WR)")
            
            return optimized
        except Exception as e:
            logger.warning(f"Could not load optimizations: {e}")
            return {}
    
    def _apply_ml_filters(self, signal: 'GenesisSignal') -> 'GenesisSignal':
        """
        Apply ML-optimized filters to signal
        
        Uses learned parameters to improve decision quality
        """
        if not self.optimized_params:
            return signal
        
        # Apply confidence threshold
        if 'min_confidence_threshold' in self.optimized_params:
            min_conf = self.optimized_params['min_confidence_threshold']
            if signal.confidence < min_conf:
                signal.execute = False
                signal.vetoes.append(f"ML: Confidence {signal.confidence:.0f}% < {min_conf:.0f}%")
        
        # Apply signal score threshold
        if 'min_signal_score' in self.optimized_params:
            min_signal = self.optimized_params['min_signal_score']
            if signal.signal_layer_score < min_signal:
                signal.execute = False
                signal.vetoes.append(f"ML: Signal score {signal.signal_layer_score:.0f} < {min_signal:.0f}")
        
        # Apply swarm threshold
        if 'min_swarm_consensus' in self.optimized_params:
            min_swarm = self.optimized_params['min_swarm_consensus']
            if signal.swarm_layer_score < min_swarm:
                signal.execute = False
                signal.vetoes.append(f"ML: Swarm score {signal.swarm_layer_score:.0f} < {min_swarm:.0f}")
        
        # Apply time filters
        if 'disable_hours' in self.optimized_params and signal.timestamp:
            # Get bad hours from recent analysis
            hour = signal.timestamp.hour
            # Simplified - in production, load from optimization history
            bad_hours = [0, 1, 2, 3, 18, 19, 20, 21, 22, 23]
            if hour in bad_hours:
                signal.execute = False
                signal.vetoes.append(f"ML: Hour {hour} historically underperforms")
        
        return signal
    
    def on_trade_close(self, trade_id: str, exit_price: float, profit_loss: float, profit_pips: float):
        """
        Called when a trade closes
        
        Records result in analytics for ML learning
        """
        try:
            self.analytics.on_trade_close(trade_id, exit_price, profit_loss, profit_pips)
            logger.info(f"📊 Trade closed: {trade_id} | P/L: ${profit_loss:.2f}")
            
            # Check if we should re-optimize
            if len(self.analytics.analyzer.trades) % 20 == 0:
                logger.info("🧠 Re-optimizing parameters based on new data...")
                self.optimized_params = self._load_optimizations()
        except Exception as e:
            logger.error(f"Error recording trade close: {e}")
    
    def generate_performance_report(self, days: int = 7) -> str:
        """Generate performance report"""
        return self.analytics.analyzer.generate_report(days)
    
    def generate_optimization_report(self, days: int = 30) -> str:
        """Generate ML optimization report"""
        return self.ml_optimizer.generate_optimization_report(days)
