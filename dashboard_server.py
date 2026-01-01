from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import zmq
import threading
import json
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Dashboard-Server")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'atl4s_secret_key'
# Switch to threading mode to avoid Eventlet/ZMQ conflicts
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ZMQ Subscriber Context
context = zmq.Context()
zmq_socket = context.socket(zmq.SUB)
zmq_socket.connect("tcp://127.0.0.1:5558") # Connect to Main Bot Publisher
zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")

import time

# Cache for latest history
latest_history = None

def zmq_listener():
    """
    Background thread to listen for ZMQ messages from the bot
    and broadcast them to WebSocket clients.
    """
    global latest_history
    logger.info("ZMQ Listener started. Waiting for data on port 5558...")
    while True:
        try:
            # Non-blocking check or just blocking? Blocking is fine in a thread.
            msg = zmq_socket.recv_string()
            
            # Format: "TOPIC JSON_DATA"
            if " " in msg:
                topic, data_str = msg.split(" ", 1)
                try:
                    data = json.loads(data_str)
                    
                    # Cache History
                    if topic == "HISTORY":
                        latest_history = data
                        logger.info(f"Cached HISTORY payload ({len(data)} candles)")

                    # Broadcast to UI
                    socketio.emit(topic, data)
                    # logger.debug(f"Broadcasted {topic}")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in ZMQ message: {msg}")
            else:
                logger.warning(f"Malformed ZMQ message: {msg}")
            
        except Exception as e:
            logger.error(f"ZMQ Error: {e}")
            time.sleep(1)

# Start ZMQ Thread
# Use native thread to avoid blocking eventlet loop with unpatched ZMQ
thread = threading.Thread(target=zmq_listener)
thread.daemon = True
thread.start()

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    if latest_history:
        emit('HISTORY', latest_history)
        logger.info("Sent cached HISTORY to new client")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/command', methods=['POST'])
def command():
    """
    Endpoint to receive commands from UI and send to Bot.
    """
    data = request.json
    cmd = data.get('command')
    logger.info(f"Received Command: {cmd}")
    # TODO: Implement ZMQ PUSH to Bot if needed
    return {'status': 'received', 'command': cmd}

if __name__ == '__main__':
    logger.info("Starting Dashboard Server on http://localhost:5000 ...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
