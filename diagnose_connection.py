import socket
import time

def test_connection():
    host = '127.0.0.1'
    port = 5557
    print(f"--- DIAGNOSTIC: Testing Connection to {host}:{port} ---")
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        print("1. Attempting Connect...")
        s.connect((host, port))
        print("2. CONNECTED! Server is listening.")
        
        msg = "TICK|TEST|1234567890|1.2345|1.2340|100\n"
        print(f"3. Sending Dummy Packet: {msg.strip()}")
        s.sendall(msg.encode('utf-8'))
        
        print("4. Packet Sent. Closing.")
        s.close()
        print("\nRESULT: SUCCESS. The Python Bridge is working correctly.")
        print("CONCLUSION: The issue is in MetaTrader 5 (Firewall/WebRequest/EA).")
        
    except ConnectionRefusedError:
        print("\nRESULT: FAILED. Connection Refused.")
        print("CONCLUSION: The Python Bridge is NOT running or Port is blocked.")
    except Exception as e:
        print(f"\nRESULT: ERROR. {e}")

if __name__ == "__main__":
    test_connection()
