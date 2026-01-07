import socket
import sys
import time
import json

HOST = "127.0.0.1"
PORT = 5557

def test_connection():
    print(f"Connecting to {HOST}:{PORT}...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        print("Connected!")
        
        # Simulate a Tick
        # TICK|SYMBOL|TIME|BID|ASK|VOL|EQUITY|POS
        msg = f"TICK|GBPUSDm|{int(time.time()*1000)}|1.25000|1.25010|100|1000.0|0\n"
        s.sendall(msg.encode('utf-8'))
        print(f"Sent: {msg.strip()}")
        
        time.sleep(1)
        s.close()
        print("Test Complete.")
    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    test_connection()
