"""Structured Logger for Trade Auditing."""

import json
import logging
import os
import time
from datetime import datetime

class StructuredLogger:
    def __init__(self, log_dir="logs/trades"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.log_file = os.path.join(log_dir, f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
    def log_trade_event(self, event_type, data):
        """Logs a trade event (ENTRY, EXIT, VETO, REFLECTION) in JSONL format."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "unixtime": time.time(),
            "event": event_type,
            "data": data
        }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logging.error(f"StructuredLogger Error: {e}")

    def log_decision_metadata(self, ticket, decision, score, confidence, details, reflection=None):
        """Captures full context of a decision."""
        payload = {
            "ticket": ticket,
            "decision": decision,
            "score": score,
            "confidence": confidence,
            "details": details,
            "reflection": reflection
        }
        self.log_trade_event("DECISION_AUDIT", payload)

# Global Instance
trade_logger = StructuredLogger()
