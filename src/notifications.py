import time
import logging
import threading

# Try to import plyer
try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False

logger = logging.getLogger("Atl4s-Notifications")

class NotificationManager:
    def __init__(self, cooldown_minutes=5):
        self.cooldown = cooldown_minutes * 60
        self.last_notification_time = 0
        self.last_signal_type = None

    def send_notification(self, title, message, signal_type=None):
        """
        Sends a notification using Plyer.
        """
        current_time = time.time()

        # Check Cooldown
        if signal_type and self.last_signal_type == signal_type:
            if current_time - self.last_notification_time < self.cooldown:
                logger.info(f"Notification suppressed (Cooldown): {title}")
                return False

        def _notify():
            try:
                if PLYER_AVAILABLE:
                    notification.notify(
                        title=title,
                        message=message,
                        app_name='Atl4s-Forex',
                        timeout=10
                    )
                else:
                    logger.warning("Plyer not found. Notification skipped. Please install plyer.")
            except Exception as e:
                logger.error(f"Plyer Notification Failed: {e}")

        # Run in a thread to avoid blocking
        t = threading.Thread(target=_notify)
        t.start()
        
        logger.info(f"Notification Triggered: {title}")
        
        # Update State
        self.last_notification_time = current_time
        if signal_type:
            self.last_signal_type = signal_type
        return True

if __name__ == "__main__":
    nm = NotificationManager(cooldown_minutes=0)
    nm.send_notification("Atl4s Test", "Plyer Notification Test")
