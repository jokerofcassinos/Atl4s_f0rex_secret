import socket
import time

HOST = '127.0.0.1'
PORT = 5557

def start_client():
    print(f"Client: Connecting to {HOST}:{PORT}...")
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((HOST, PORT))
        data = client.recv(1024)
        print(f"Client: Received: {data.decode()}")
        client.close()
    except Exception as e:
        print(f"Client Error: {e}")

if __name__ == "__main__":
    time.sleep(1) # Wait for server
    start_client()
