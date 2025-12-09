import socket
import logging
import config
import json
import threading
import time

logger = logging.getLogger("Atl4s-Bridge")

class ZmqBridge: # Keeping name for compatibility, but it's now SocketBridge
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = config.ZMQ_REQ_PORT # Use one port for bidirectional comms
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.setblocking(False)
        
        self.conn = None
        self.addr = None
        self.running = True
        self.buffer = ""
        
        # Start connection listener thread
        self.thread = threading.Thread(target=self._accept_connections)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Socket Bridge Initialized. Listening on {self.host}:{self.port}")

    def _accept_connections(self):
        logger.info("Bridge: Accept Thread Started.")
        while self.running:
            try:
                if self.conn is None:
                    try:
                        # logger.debug("Bridge: Waiting for connection...")
                        conn, addr = self.server_socket.accept()
                        conn.setblocking(False)
                        self.conn = conn
                        self.addr = addr
                        logger.info(f"MT5 Connected: {addr}")
                    except BlockingIOError:
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error(f"Bridge: Accept Error: {e}")
                        time.sleep(1)
                else:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Bridge: Thread Error: {e}")
                time.sleep(1)
        logger.info("Bridge: Accept Thread Stopped.")

    def get_tick(self):
        """
        Reads data from socket buffer.
        Returns parsed tick dict or None.
        """
        if not self.conn:
            return None
            
        try:
            # Read buffer
            try:
                data = self.conn.recv(4096)
                if not data:
                    self.conn = None # Disconnected
                    return None
                
                self.buffer += data.decode('utf-8')
                
                # Process complete messages
                last_tick = None
                
                while '\n' in self.buffer:
                    msg, self.buffer = self.buffer.split('\n', 1)
                    msg = msg.strip()
                    if not msg: continue
                    
                    parts = msg.split('|')
                    if parts[0] == "TICK":
                        if len(parts) < 7:
                            logger.warning(f"Malformed Tick: {msg}")
                            continue
                            
                        last_tick = {
                            "type": "TICK",
                            "symbol": parts[1],
                            "time": int(parts[2]),
                            "bid": float(parts[3]),
                            "ask": float(parts[4]),
                            "last": float(parts[5]),
                            "volume": int(parts[6])
                        }
                        
                        # Fix for Forex symbols where Last price is 0
                        if last_tick['last'] <= 0:
                            if last_tick['bid'] > 0:
                                last_tick['last'] = last_tick['bid']
                            else:
                                logger.warning(f"Received Zero Price Tick (Bid also 0): {msg}")
                            
                return last_tick
                
            except BlockingIOError:
                return None # No data
                
        except Exception as e:
            logger.error(f"Socket Receive Error: {e}")
            self.conn = None
            return None

    def send_command(self, command, params=None):
        """
        Send a command to MT5.
        """
        if not self.conn:
            logger.warning("Cannot send command: MT5 Disconnected")
            return "ERROR"
            
        try:
            cmd_str = command
            if params:
                cmd_str += "|" + "|".join([str(p) for p in params])
            
            cmd_str += "\n" # Delimiter
            
            logger.info(f"Sending Command: {cmd_str.strip()}")
            self.conn.sendall(cmd_str.encode('utf-8'))
            
            # Wait for reply (simplified, assuming immediate ACK)
            # In a robust system, we'd use a request ID.
            # For now, we assume the next message is the reply if it's not a TICK.
            # But since get_tick reads everything, this is tricky.
            # For this simplified version, we just send and assume execution.
            return "SENT"
            
        except Exception as e:
            logger.error(f"Socket Send Error: {e}")
            self.conn = None
            return "ERROR"

    def close(self):
        self.running = False
        if self.conn:
            self.conn.close()
        self.server_socket.close()
        logger.info("Socket Bridge Closed.")
