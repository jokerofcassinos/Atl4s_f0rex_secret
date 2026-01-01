from src.notifications import NotificationManager
import time

print("Testing Windows Notification...")
nm = NotificationManager(cooldown_minutes=0)
nm.send_notification("Atl4s-Forex", "System Check: Notifications are working.")
print("Command sent. Please check your system tray/action center.")
time.sleep(2)
print("Done.")
