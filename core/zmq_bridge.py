
import logging
import socket
import threading
import time
import json

logger = logging.getLogger("ZmqBridge")

class ZmqBridge:
    """
    Native Socket Bridge (Replacing ZMQ).
    Acts as a TCP Server. Multiplexes MQL5 Clients.
    """
    def __init__(self, port=5557):
        self.port = port
        self.host = "0.0.0.0"
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5) # Allow backlog
            self.server_socket.setblocking(False) # Non-blocking accept
            logger.info(f"Bridge Server started on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to bind port {self.port}: {e}")

        # Client Registry: Symbol -> Socket
        self.clients = {} 
        self.socket_map = {} # Socket -> Symbol (reverse lookup)
        
        self.conn_lock = threading.Lock()
        
        self.latest_tick = None
        self.latest_trades = [] # Store open trades buffer
        self.buffer = ""
        self.running = True
        
        # We need to handle multiple clients, so a single buffer won't work perfectly if interleaved.
        # But MQL5 sends whole lines mostly.
        # Ideally we spawn a thread per client.
        
        # Background Accept Loop
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()

    def _server_loop(self):
        # We use select or just simple loop for now since we expect < 5 clients
        # Actually, handling multiple sockets in one thread without select is hard.
        # Let's use a simple per-client thread approach.
        
        while self.running:
            try:
                client, addr = self.server_socket.accept()
                logger.info(f"MQL5 Client Connected: {addr}")
                client.setblocking(True) # Simplify reading in thread
                
                # Spawn Handler Thread
                ct = threading.Thread(target=self._client_handler, args=(client, addr), daemon=True)
                ct.start()
                
            except BlockingIOError:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Accept Error: {e}")
                time.sleep(1)

    def _client_handler(self, sock, addr):
        """
        Handles a single MQL5 connection.
        """
        # We don't know the symbol yet.
        detected_symbol = None
        buffer = ""
        
        try:
            while self.running:
                data = sock.recv(4096)
                if not data: break
                
                text = data.decode('utf-8', errors='ignore')
                buffer += text
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line: continue
                    
                    # Parse
                    fields = self._parse_line(line)
                    
                    # Process TRADES_JSON separately (no symbol field)
                    if fields and fields.get('type') == 'TRADES_JSON':
                        self.latest_trades = fields.get('trades', [])
                        logger.debug(f"Received Trades Update: {len(self.latest_trades)} trades")
                        continue
                    
                    if fields and 'symbol' in fields:
                        sym = fields['symbol']
                        
                        # Register if new
                        if detected_symbol != sym:
                            detected_symbol = sym
                            with self.conn_lock:
                                self.clients[sym] = sock
                                self.socket_map[sock] = sym
                                logger.info(f"Registered Socket for {sym}")
                                
                        # Process Data (Update internal state)
                        if fields.get('type') == 'TICK':
                            self.latest_tick = fields
                            
        except ConnectionResetError:
            logger.warning(f"Connection Reset: {detected_symbol}")
        except Exception as e:
            logger.error(f"Client Error {detected_symbol}: {e}")
        finally:
            with self.conn_lock:
                if detected_symbol and detected_symbol in self.clients:
                    del self.clients[detected_symbol]
                if sock in self.socket_map:
                    del self.socket_map[sock]
            sock.close()
            logger.info(f"Detailed Disconnect: {detected_symbol}")

    def _parse_line(self, line):
        # TICK|SYMBOL|TIME|BID|ASK|VOL|EQUITY|POS
        try:
            if line.startswith("TICK"):
                parts = line.split('|')
                if len(parts) >= 6:
                    tick = {
                        'type': 'TICK',
                        'symbol': parts[1],
                        'time': int(parts[2]),
                        'bid': float(parts[3]),
                        'ask': float(parts[4]),
                        'volume': int(parts[5])
                    }
                    if len(parts) >= 8:
                        tick['equity'] = float(parts[6])
                        tick['positions'] = int(parts[7])
                    if len(parts) >= 9:
                        tick['profit'] = float(parts[8])
                    if len(parts) >= 11:
                        tick['best_profit'] = float(parts[9])
                        tick['best_ticket'] = int(parts[10])
                    return tick
                    
            elif line.startswith("TRADES_JSON"):
                # Format: TRADES_JSON|{json_string}
                try:
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        data = json.loads(parts[1]) # List of dicts
                        return {
                            'type': 'TRADES_JSON',
                            'trades': data
                        }
                except Exception as e:
                    logger.error(f"JSON Parse Error: {e}")
                    return None
                    
            return None
        except:
            return None

    def get_tick(self):
        # This is simple/naive (last tick from *any* symbol)
        # Main.py might need to filter.
        t = self.latest_tick
        if t:
            t['trades_json'] = self.latest_trades
        return t

    def send_command(self, action, params=None):
        """
        Sends command to MQL5.
        Tries to route to the correct symbol if 'symbol' is in params.
        """
        msg = f"{action}"
        if params:
            msg += "|" + "|".join(map(str, params))
        msg += "\n"
        encoded = msg.encode('utf-8')
        
        # Determine Target
        target_sock = None
        target_sym = None
        
        # Extract symbol from params?
        # Params is a list usually? Or dict?
        # In main.py: bridge.send_command("ORDER", [side, symbol, volume...])
        # So params[1] is symbol?
        # Let's assume params is a list and check if any element matches a known symbol.
        
        with self.conn_lock:
            # Broadcast to ALL for critical commands? No, duplicate orders.
            # Try to match symbol
            if params:
                for p in params:
                    if str(p) in self.clients:
                        target_sock = self.clients[str(p)]
                        target_sym = str(p)
                        break
            
            # Fallback: If only 1 client, use it.
            if not target_sock and len(self.clients) == 1:
                target_sock = list(self.clients.values())[0]
                target_sym = "SINGLE_FALLBACK"

            if target_sock:
                try:
                    target_sock.sendall(encoded)
                    # logger.info(f"Sent {action} to {target_sym}")
                except Exception as e:
                    logger.error(f"Send Error to {target_sym}: {e}")
            else:
                 logger.warning(f"Could not route command '{action}' (Clients: {list(self.clients.keys())})")

    def send_dashboard(self, state):
        pass

    # --- VISUALIZATION COMMANDS ---
    def send_draw_rect(self, symbol, name, p1, p2, t1, t2, color):
        # DRAW_RECT|NAME|P1|P2|TIME1|TIME2|COLOR
        # Need to route to symbol
        self.send_command("DRAW_RECT", [symbol, name, p1, p2, int(t1), int(t2), color])

    def send_draw_line(self, symbol, name, p1, p2, t1, t2, color):
        # DRAW_LINE|NAME|P1|P2|TIME1|TIME2|COLOR
        self.send_command("DRAW_LINE", [symbol, name, p1, p2, int(t1), int(t2), color])

    def send_draw_text(self, symbol, name, price, time, text, color):
        # DRAW_TEXT|NAME|PRICE|TIME|TEXT|COLOR
        self.send_command("DRAW_TEXT", [symbol, name, price, int(time), text, color])
