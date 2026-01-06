"""
AGI Ultra-Complete: Unified Communication Layer

Camada de comunicação unificada entre componentes:
- EventBus: Pub/Sub inteligente
- SharedMemoryLayer: Memória compartilhada
- UnifiedProtocol: Protocolo padronizado
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import queue

logger = logging.getLogger("UnifiedCommunication")


class MessagePriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class MessageType(Enum):
    EVENT = "event"
    COMMAND = "command"
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"


@dataclass
class Message:
    """Message in the communication system."""
    id: str
    type: MessageType
    source: str
    target: str
    payload: Any
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = 60.0


@dataclass
class Subscription:
    """Event subscription."""
    subscriber_id: str
    event_type: str
    callback: Callable
    filter_func: Optional[Callable] = None


class EventBus:
    """Intelligent event bus with pub/sub."""
    
    def __init__(self):
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=1000)
        self.message_queue: queue.PriorityQueue = queue.PriorityQueue()
        
        self._message_counter = 0
        self._lock = threading.Lock()
        
        logger.info("EventBus initialized")
    
    def subscribe(self, subscriber_id: str, event_type: str, 
                  callback: Callable, filter_func: Callable = None):
        """Subscribe to events."""
        sub = Subscription(
            subscriber_id=subscriber_id,
            event_type=event_type,
            callback=callback,
            filter_func=filter_func
        )
        
        with self._lock:
            self.subscriptions[event_type].append(sub)
        
        logger.debug(f"Subscribed {subscriber_id} to {event_type}")
    
    def unsubscribe(self, subscriber_id: str, event_type: str = None):
        """Unsubscribe from events."""
        with self._lock:
            if event_type:
                self.subscriptions[event_type] = [
                    s for s in self.subscriptions[event_type]
                    if s.subscriber_id != subscriber_id
                ]
            else:
                for et in self.subscriptions:
                    self.subscriptions[et] = [
                        s for s in self.subscriptions[et]
                        if s.subscriber_id != subscriber_id
                    ]
    
    def publish(self, event_type: str, payload: Any, 
                source: str = "system", priority: MessagePriority = MessagePriority.NORMAL):
        """Publish an event."""
        self._message_counter += 1
        
        message = Message(
            id=f"evt_{self._message_counter}",
            type=MessageType.EVENT,
            source=source,
            target="broadcast",
            payload=payload,
            priority=priority
        )
        
        self.event_history.append(message)
        
        with self._lock:
            subs = self.subscriptions.get(event_type, [])
            subs += self.subscriptions.get("*", [])
        
        for sub in subs:
            try:
                if sub.filter_func is None or sub.filter_func(payload):
                    sub.callback(message)
            except Exception as e:
                logger.error(f"Error in subscriber {sub.subscriber_id}: {e}")
    
    def get_history(self, event_type: str = None, limit: int = 100) -> List[Message]:
        """Get event history."""
        history = list(self.event_history)
        
        if event_type:
            history = [m for m in history if m.payload.get('type') == event_type]
        
        return history[-limit:]


class SharedMemoryLayer:
    """Shared memory between components."""
    
    def __init__(self):
        self.memory: Dict[str, Dict] = {}
        self.versions: Dict[str, int] = defaultdict(int)
        self.access_log: deque = deque(maxlen=500)
        self._lock = threading.Lock()
        
        logger.info("SharedMemoryLayer initialized")
    
    def set(self, namespace: str, key: str, value: Any, ttl_seconds: float = None):
        """Set a value in shared memory."""
        with self._lock:
            if namespace not in self.memory:
                self.memory[namespace] = {}
            
            self.memory[namespace][key] = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl_seconds,
                'version': self.versions[f"{namespace}.{key}"]
            }
            
            self.versions[f"{namespace}.{key}"] += 1
        
        self.access_log.append({
            'operation': 'set',
            'namespace': namespace,
            'key': key,
            'timestamp': time.time()
        })
    
    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """Get a value from shared memory."""
        with self._lock:
            if namespace not in self.memory:
                return default
            
            entry = self.memory[namespace].get(key)
            if entry is None:
                return default
            
            if entry['ttl'] is not None:
                age = time.time() - entry['timestamp']
                if age > entry['ttl']:
                    del self.memory[namespace][key]
                    return default
            
            return entry['value']
        
        self.access_log.append({
            'operation': 'get',
            'namespace': namespace,
            'key': key,
            'timestamp': time.time()
        })
    
    def get_all(self, namespace: str) -> Dict[str, Any]:
        """Get all values in namespace."""
        with self._lock:
            if namespace not in self.memory:
                return {}
            
            return {k: v['value'] for k, v in self.memory[namespace].items()}
    
    def delete(self, namespace: str, key: str = None):
        """Delete from shared memory."""
        with self._lock:
            if namespace in self.memory:
                if key:
                    self.memory[namespace].pop(key, None)
                else:
                    del self.memory[namespace]
    
    def get_version(self, namespace: str, key: str) -> int:
        """Get version of a key."""
        return self.versions.get(f"{namespace}.{key}", 0)


class ComponentRegistry:
    """Registry of all AGI components."""
    
    def __init__(self):
        self.components: Dict[str, Dict] = {}
        self.health_status: Dict[str, str] = {}
        
        logger.info("ComponentRegistry initialized")
    
    def register(self, component_id: str, component: Any, metadata: Dict = None):
        """Register a component."""
        self.components[component_id] = {
            'instance': component,
            'metadata': metadata or {},
            'registered_at': time.time()
        }
        self.health_status[component_id] = 'healthy'
        
        logger.info(f"Registered component: {component_id}")
    
    def unregister(self, component_id: str):
        """Unregister a component."""
        self.components.pop(component_id, None)
        self.health_status.pop(component_id, None)
    
    def get(self, component_id: str) -> Optional[Any]:
        """Get a component."""
        entry = self.components.get(component_id)
        return entry['instance'] if entry else None
    
    def get_all(self) -> List[str]:
        """Get all component IDs."""
        return list(self.components.keys())
    
    def update_health(self, component_id: str, status: str):
        """Update component health status."""
        if component_id in self.health_status:
            self.health_status[component_id] = status
    
    def get_healthy_components(self) -> List[str]:
        """Get all healthy components."""
        return [c for c, s in self.health_status.items() if s == 'healthy']


class UnifiedCommunicationLayer:
    """Main unified communication layer."""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.shared_memory = SharedMemoryLayer()
        self.registry = ComponentRegistry()
        
        self.message_handlers: Dict[str, Callable] = {}
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0
        }
        
        logger.info("UnifiedCommunicationLayer initialized")
    
    def register_component(self, component_id: str, component: Any):
        """Register a component."""
        self.registry.register(component_id, component)
        
        if hasattr(component, 'subscribe_events'):
            events = component.subscribe_events()
            for event, callback in events.items():
                self.event_bus.subscribe(component_id, event, callback)
    
    def send_message(self, source: str, target: str, payload: Any,
                     msg_type: MessageType = MessageType.COMMAND,
                     priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """Send a message between components."""
        self.stats['messages_sent'] += 1
        
        target_component = self.registry.get(target)
        if target_component is None:
            logger.warning(f"Target component not found: {target}")
            return False
        
        if hasattr(target_component, 'receive_message'):
            try:
                message = Message(
                    id=f"msg_{self.stats['messages_sent']}",
                    type=msg_type,
                    source=source,
                    target=target,
                    payload=payload,
                    priority=priority
                )
                target_component.receive_message(message)
                self.stats['messages_received'] += 1
                return True
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                self.stats['errors'] += 1
                return False
        
        return False
    
    def broadcast(self, source: str, event_type: str, payload: Any):
        """Broadcast an event."""
        self.event_bus.publish(event_type, payload, source)
    
    def share_state(self, component_id: str, key: str, value: Any):
        """Share state via shared memory."""
        self.shared_memory.set(component_id, key, value)
    
    def get_shared_state(self, component_id: str, key: str) -> Any:
        """Get shared state."""
        return self.shared_memory.get(component_id, key)
    
    def get_status(self) -> Dict[str, Any]:
        """Get communication layer status."""
        return {
            'components': len(self.registry.components),
            'healthy_components': len(self.registry.get_healthy_components()),
            'subscriptions': sum(len(s) for s in self.event_bus.subscriptions.values()),
            'stats': self.stats
        }


communication_layer = UnifiedCommunicationLayer()
