
import logging
import asyncio
import json

try:
    import pynng
    HAS_NNG = True
except ImportError:
    HAS_NNG = False

logger = logging.getLogger("QuantumBus")

class QuantumBus:
    """
    The Synapse.
    Uses NNG (Nanomsg Next Gen) for microsecond IPC.
    Architecture: Pub/Sub Model.
    """
    def __init__(self, mode="PUB", address="ipc:///tmp/omega_synapse"):
        self.mode = mode
        self.address = address
        self.socket = None
        self.fallback_queue = asyncio.Queue() if not HAS_NNG else None
        
        if HAS_NNG:
            self._setup_nng()
        else:
            logger.warning("QuantumBus: NNG not found. Using Asyncio Queue (Slower/Thread-locked).")

    def _setup_nng(self):
        try:
            if self.mode == "PUB":
                self.socket = pynng.Pub0()
                self.socket.listen(self.address)
            elif self.mode == "SUB":
                self.socket = pynng.Sub0()
                self.socket.subscribe(b'')
                self.socket.dial(self.address)
        except Exception as e:
            logger.error(f"NNG Init Error: {e}")
            self.socket = None

    async def publish(self, topic: str, data: dict):
        payload = json.dumps({'topic': topic, 'data': data}).encode('utf-8')
        
        if self.socket:
            try:
                await self.socket.asend(payload)
            except Exception as e:
                logger.error(f"Pub Error: {e}")
        else:
            await self.fallback_queue.put(payload)

    async def receive(self):
        """Yields messages."""
        while True:
            if self.socket:
                try:
                    msg = await self.socket.arecv()
                    yield json.loads(msg.decode('utf-8'))
                except pynng.Timeout:
                    continue
            else:
                msg = await self.fallback_queue.get()
                yield json.loads(msg.decode('utf-8'))
                
    def close(self):
        if self.socket:
            self.socket.close()
