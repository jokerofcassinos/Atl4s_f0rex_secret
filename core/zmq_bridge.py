
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
    def __init__(self, port=5558):
        self.port = port
        self.host = "0.0.0.0"
        self.running = True
        
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.setblocking(False)
            logger.info(f"Bridge Server started on {self.host}:{self.port}")
        except Exception as e:
            logger.critical(f"FATAL: FAILED TO BIND PORT {self.port}: {e}")
            raise SystemExit("ZOMBIE_PORT_CONFLICT")

        self.conn_lock = threading.Lock()
        self.clients = {}  # Symbol -> Socket
        self.socket_map = {} # Socket -> Symbol
        
        self.latest_tick = None
        self.latest_trades = {}  # Symbol -> List of Trades
        self.global_latest_trades = []
        
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()

    def _server_loop(self):
        while self.running:
            try:
                client, addr = self.server_socket.accept()
                logger.info(f"MQL5 Client Connected: {addr}")
                client.setblocking(False)
                ct = threading.Thread(target=self._client_handler, args=(client, addr), daemon=True)
                ct.start()
            except BlockingIOError:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Accept Error: {e}")
                time.sleep(1)

    def _client_handler(self, sock, addr):
        detected_symbol = None
        buffer = ""
        try:
            while self.running:
                try:
                    data = sock.recv(4096)
                except BlockingIOError:
                    time.sleep(0.01)
                    continue
                if not data: break
                
                buffer += data.decode('utf-8', errors='ignore')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line: continue
                    
                    fields = self._parse_line(line)
                    if not fields: continue

                    if fields.get('type') == 'TRADES_JSON':
                        trades = fields.get('trades', [])
                        sym = fields.get('symbol')
                        if sym:
                            self.latest_trades[sym] = trades
                            # If we don't have a detected_symbol yet, use this one
                            if not detected_symbol:
                                detected_symbol = sym
                                with self.conn_lock:
                                    self.clients[sym] = sock
                                    self.socket_map[sock] = sym
                        else:
                            # Fallback for old format
                            if detected_symbol:
                                self.latest_trades[detected_symbol] = trades
                            self.global_latest_trades = trades
                        continue
                    
                    if 'symbol' in fields:
                        sym = fields['symbol']
                        if detected_symbol != sym:
                            detected_symbol = sym
                            with self.conn_lock:
                                self.clients[sym] = sock
                                self.socket_map[sock] = sym
                                logger.info(f"Registered Socket for {sym}")
                        
                        if fields.get('type') == 'TICK':
                            self.latest_tick = fields
                            
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
                    if len(parts) >= 8: tick['equity'], tick['positions'] = float(parts[6]), int(parts[7])
                    if len(parts) >= 9: tick['profit'] = float(parts[8])
                    if len(parts) >= 11:
                        tick['best_profit'] = float(parts[9]) if float(parts[9]) > -990000 else 0.0
                        tick['best_ticket'] = int(parts[10])
                    if len(parts) >= 14: tick['indicator_val'] = float(parts[12])
                    return tick
            elif line.startswith("TRADES_JSON"):
                parts = line.split('|')
                if len(parts) == 3:
                    # New Format: TRADES_JSON|SYMBOL|[...]
                    return {'type': 'TRADES_JSON', 'symbol': parts[1], 'trades': json.loads(parts[2])}
                elif len(parts) == 2:
                    # Old Format: TRADES_JSON|[...]
                    return {'type': 'TRADES_JSON', 'trades': json.loads(parts[1])}
        except Exception as e:
            logger.error(f"Parse Error: {e}")
        return None

    def get_tick(self):
        """Returns the latest tick + injected trades for that symbol"""
        tick = self.latest_tick.copy() if self.latest_tick else None
        if tick:
            sym = tick.get('symbol')
            tick['trades_json'] = self.latest_trades.get(sym, self.global_latest_trades)
        return tick

    def send_command(self, action, params=None):
        if "DRAW" in action: 
            # Allow DRAW commands now that we have wrapper methods
            pass
            # previously: return # Logic to skip visuals
        msg = f"{action}"
        if params: msg += "|" + "|".join(map(str, params))
        msg += "\n"
        encoded = msg.encode('utf-8')
        
        target_sock = None
        target_sym = None
        
        # Routing logic
        with self.conn_lock:
            if params and len(params) > 0:
                cand = str(params[0])
                if cand in self.clients:
                    target_sock = self.clients[cand]
                    target_sym = cand
                else:
                    # Fuzzy match
                    for client_sym, sock in self.clients.items():
                        if client_sym.startswith(cand) or cand.startswith(client_sym):
                            target_sock, target_sym = sock, client_sym
                            break
            
            if not target_sock and len(self.clients) == 1:
                target_sym = list(self.clients.keys())[0]
                target_sock = self.clients[target_sym]

            # Broadcast Mode
            if "CLOSE_ALL" in action or "PRUNE" in action:
                if not target_sock:
                    for s in self.clients.values(): s.sendall(encoded)
                    return

        if target_sock:
            try:
                target_sock.sendall(encoded)
                logger.info(f"BRIDGE TX -> {target_sym}: {action}")
            except Exception as e:
                logger.error(f"Send Error to {target_sym}: {e}")

    def get_open_trades(self):
        # Flatten for convenience of legacy calls
        all_trades = []
        for trades in self.latest_trades.values():
            all_trades.extend(trades)
        return all_trades if all_trades else self.global_latest_trades

    def send_draw_line(self, symbol, name, price1, price2, time1, time2, color):
        """Sends a DRAW_LINE command to MT5."""
        # Command: DRAW_LINE|symbol|name|price1|price2|time1|time2|color
        self.send_command("DRAW_LINE", [symbol, name, price1, price2, time1, time2, color])

    def send_draw_text(self, symbol, name, price, time, text, color):
        """Sends a DRAW_TEXT command to MT5."""
        # Command: DRAW_TEXT|symbol|name|price|time|text|color
        self.send_command("DRAW_TEXT", [symbol, name, price, time, text, color])
