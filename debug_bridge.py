import socket
import time

HOST = '0.0.0.0' # Listen on all interfaces
PORT = 5557

def start_server():
    print(f"Server: Binding to {HOST}:{PORT}...")
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((HOST, PORT))
        server.listen(1)
        print("Server: Listening...")
        
        while True:
            print("Server: Waiting for connection...")
            conn, addr = server.accept()
            print(f"Server: Connected by {addr}")
            
            try:
                conn.sendall(b"HELLO_FROM_PYTHON\n")
                
                # Keep connection open and read data
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    print(f"Server: Received {len(data)} bytes: {data.decode().strip()}")
            except Exception as e:
                print(f"Server: Connection Error: {e}")
            finally:
                conn.close()
                print("Server: Connection closed. Waiting for next...")
            
    except Exception as e:
        print(f"Server Error: {e}")

if __name__ == "__main__":
    start_server()
