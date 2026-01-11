"""
Helper method for Genesis to convert LaplacePrediction to SignalContext
Add this to main_genesis.py GenesisSystem class
"""

def _convert_laplace_signal(self, laplace_signal):
    """Convert LaplacePrediction to SignalContext format"""
    
    if not laplace_signal:
        return SignalContext(
            smc={'trend': 'RANGING', 'active_order_blocks': [], 'active_fvgs': [], 'entry_signal': {'direction': None}},
            m8_fib={'signal': 'WAIT', 'confidence': 0},
            quarterly={'tradeable': False, 'phase': 'Q1', 'score': 0},
            momentum={'score': 0, 'trend': 'NEUTRAL'},
            volatility={'regime': 'NORMAL', 'score': 0}
        )
    
    # Map LaplacePrediction to SignalContext
    return SignalContext(
        smc={
            'trend': 'BULLISH' if laplace_signal.direction == 'BUY' else 'BEARISH' if laplace_signal.direction == 'SELL' else 'RANGING',
            'active_order_blocks': [],
            'active_fvgs': [],
            'entry_signal': {
                'direction': laplace_signal.direction,
                'confidence': laplace_signal.confidence
            }
        },
        m8_fib={
            'signal': laplace_signal.direction,
            'confidence': laplace_signal.confidence,
            'gate': 'Q3'  # Assume golden zone if executing
        },
        quarterly={
            'tradeable': laplace_signal.execute,
            'phase': 'Q3',
            'score': 5 if laplace_signal.execute else 0,
            'reason': laplace_signal.primary_signal
        },
        momentum={
            'score': laplace_signal.confidence,
            'trend': laplace_signal.direction if laplace_signal.execute else 'NEUTRAL'
        },
        volatility={
            'regime': 'NORMAL',
            'score': laplace_signal.confidence
        }
    )
