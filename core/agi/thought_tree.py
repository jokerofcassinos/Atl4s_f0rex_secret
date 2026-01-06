"""
Fase 6: Sistema de Árvore de Pensamento
Thought Tree - Estrutura de pensamento recursivo para módulos

AGI Ultra: Expanded with ThoughtGraph, ConfidencePropagator, TreeCompressor,
and incremental persistence for massive scale thought processing.

Cada módulo cria sua própria árvore de pensamento, e todas se interligam
para formar uma rede de raciocínio global.
"""

import logging
import json
import os
import pickle
import threading
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import uuid

logger = logging.getLogger("ThoughtTree")

@dataclass
class ThoughtNode:
    """
    Um nó na árvore de pensamento.
    Cada nó representa uma pergunta, resposta, ou decisão.
    """
    node_id: str
    module_name: str
    question: str
    answer: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)  # IDs de nós relacionados
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child_id: str):
        """Adiciona um nó filho."""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def add_connection(self, node_id: str):
        """Adiciona uma conexão com outro nó."""
        if node_id not in self.connections:
            self.connections.append(node_id)

class ThoughtTree:
    """
    Árvore de pensamento de um módulo.
    Cada módulo tem sua própria árvore que pode se interligar com outras.
    """
    
    def __init__(self, module_name: str, max_depth: int = 10):
        """
        Inicializa uma árvore de pensamento.
        
        Args:
            module_name: Nome do módulo que possui esta árvore
            max_depth: Profundidade máxima da árvore (evita loops infinitos)
        """
        self.module_name = module_name
        self.max_depth = max_depth
        self.nodes: Dict[str, ThoughtNode] = {}
        self.root_nodes: List[str] = []  # IDs dos nós raiz
        self._max_depth_warned = False  # Flag to only warn once
        
    def create_node(self, question: str, parent_id: Optional[str] = None, 
                   context: Optional[Dict[str, Any]] = None,
                   confidence: float = 0.0) -> str:
        """
        Cria um novo nó na árvore.
        
        Args:
            question: Pergunta que este nó representa
            parent_id: ID do nó pai (None para nó raiz)
            context: Contexto adicional
            confidence: Confiança na resposta (0-1)
            
        Returns:
            ID do nó criado
        """
        node_id = str(uuid.uuid4())
        
        # Verifica profundidade
        depth = 0
        if parent_id:
            depth = self._calculate_depth(parent_id)
            if depth >= self.max_depth:
                if not self._max_depth_warned:
                    logger.debug(f"ThoughtTree {self.module_name}: Max depth ({self.max_depth}) reached.")
                    self._max_depth_warned = True
                return parent_id  # Retorna o pai para evitar criar nó
        
        node = ThoughtNode(
            node_id=node_id,
            module_name=self.module_name,
            question=question,
            context=context or {},
            confidence=confidence,
            parent_id=parent_id
        )
        
        self.nodes[node_id] = node
        
        # Se é nó raiz, adiciona à lista
        if parent_id is None:
            self.root_nodes.append(node_id)
        else:
            # Adiciona como filho do pai
            if parent_id in self.nodes:
                self.nodes[parent_id].add_child(node_id)
        
        return node_id
    
    def answer_node(self, node_id: str, answer: str, confidence: float = 0.0):
        """
        Responde a um nó (pergunta).
        
        Args:
            node_id: ID do nó
            answer: Resposta à pergunta
            confidence: Confiança na resposta
        """
        if node_id in self.nodes:
            self.nodes[node_id].answer = answer
            self.nodes[node_id].confidence = confidence
            self.nodes[node_id].timestamp = datetime.now()
        else:
            logger.warning(f"ThoughtTree {self.module_name}: Node {node_id} not found.")
    
    def connect_nodes(self, node_id1: str, node_id2: str):
        """
        Conecta dois nós (cria relação entre pensamentos).
        
        Args:
            node_id1: ID do primeiro nó
            node_id2: ID do segundo nó
        """
        if node_id1 in self.nodes and node_id2 in self.nodes:
            self.nodes[node_id1].add_connection(node_id2)
            self.nodes[node_id2].add_connection(node_id1)
        else:
            logger.warning(f"ThoughtTree {self.module_name}: Cannot connect nodes. One or both not found.")
    
    def _calculate_depth(self, node_id: str) -> int:
        """Calcula a profundidade de um nó."""
        depth = 0
        current_id = node_id
        
        while current_id and current_id in self.nodes:
            parent_id = self.nodes[current_id].parent_id
            if parent_id is None:
                break
            depth += 1
            current_id = parent_id
        
        return depth
    
    def get_thought_chain(self, node_id: str) -> List[ThoughtNode]:
        """
        Obtém a cadeia de pensamento até um nó (do raiz até o nó).
        
        Args:
            node_id: ID do nó final
            
        Returns:
            Lista de nós da raiz até o nó especificado
        """
        chain = []
        current_id = node_id
        
        while current_id and current_id in self.nodes:
            chain.insert(0, self.nodes[current_id])
            current_id = self.nodes[current_id].parent_id
        
        return chain
    
    def get_recent_thoughts(self, limit: int = 10) -> List[ThoughtNode]:
        """
        Obtém os pensamentos mais recentes.
        
        Args:
            limit: Número máximo de pensamentos
            
        Returns:
            Lista de nós ordenados por timestamp (mais recentes primeiro)
        """
        all_nodes = list(self.nodes.values())
        all_nodes.sort(key=lambda n: n.timestamp, reverse=True)
        return all_nodes[:limit]
    
    def get_unanswered_questions(self) -> List[ThoughtNode]:
        """
        Obtém todas as perguntas sem resposta.
        
        Returns:
            Lista de nós sem resposta
        """
        return [node for node in self.nodes.values() if node.answer is None]

class GlobalThoughtOrchestrator:
    """
    Orquestrador global de árvores de pensamento.
    Gerencia todas as árvores de todos os módulos e as interliga.
    """
    
    def __init__(self):
        self.trees: Dict[str, ThoughtTree] = {}
        self.cross_module_connections: Dict[str, List[str]] = {}  # node_id -> [other_module_node_ids]
        
    def get_or_create_tree(self, module_name: str) -> ThoughtTree:
        """
        Obtém ou cria uma árvore para um módulo.
        
        Args:
            module_name: Nome do módulo
            
        Returns:
            ThoughtTree do módulo
        """
        if module_name not in self.trees:
            self.trees[module_name] = ThoughtTree(module_name)
        return self.trees[module_name]
    
    def connect_cross_module(self, module1: str, node_id1: str, 
                            module2: str, node_id2: str):
        """
        Conecta nós de diferentes módulos.
        
        Args:
            module1: Nome do primeiro módulo
            node_id1: ID do nó no primeiro módulo
            module2: Nome do segundo módulo
            node_id2: ID do nó no segundo módulo
        """
        if module1 not in self.trees or module2 not in self.trees:
            logger.warning("Cannot connect cross-module: One or both modules not found.")
            return
        
        # Adiciona conexão cruzada
        key = f"{module1}:{node_id1}"
        if key not in self.cross_module_connections:
            self.cross_module_connections[key] = []
        
        connection = f"{module2}:{node_id2}"
        if connection not in self.cross_module_connections[key]:
            self.cross_module_connections[key].append(connection)
    
    def find_similar_thoughts(self, module_name: str, question: str, 
                             threshold: float = 0.7) -> List[tuple]:
        """
        Encontra pensamentos similares em outros módulos.
        
        Args:
            module_name: Módulo que está perguntando
            question: Pergunta a buscar
            threshold: Limiar de similaridade (0-1)
            
        Returns:
            Lista de tuplas (module_name, node_id, similarity_score)
        """
        # Implementação simplificada: busca por palavras-chave
        # Em produção, usar embedding vetorial ou NLP avançado
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        similar = []
        
        for mod_name, tree in self.trees.items():
            if mod_name == module_name:
                continue  # Pula o próprio módulo
            
            for node_id, node in tree.nodes.items():
                if node.question:
                    node_question_lower = node.question.lower()
                    node_words = set(node_question_lower.split())
                    
                    # Similaridade simples: Jaccard
                    intersection = len(question_words & node_words)
                    union = len(question_words | node_words)
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity >= threshold:
                        similar.append((mod_name, node_id, similarity))
        
        # Ordena por similaridade
        similar.sort(key=lambda x: x[2], reverse=True)
        return similar
    
    def get_global_thought_summary(self) -> Dict[str, Any]:
        """
        Obtém um resumo global de todas as árvores de pensamento.
        
        Returns:
            Dicionário com estatísticas globais
        """
        total_nodes = sum(len(tree.nodes) for tree in self.trees.values())
        total_connections = sum(len(conns) for conns in self.cross_module_connections.values())
        unanswered = sum(len(tree.get_unanswered_questions()) for tree in self.trees.values())
        
        return {
            'total_modules': len(self.trees),
            'total_nodes': total_nodes,
            'total_cross_module_connections': total_connections,
            'unanswered_questions': unanswered,
            'modules': {
                name: {
                    'nodes': len(tree.nodes),
                    'root_nodes': len(tree.root_nodes),
                    'unanswered': len(tree.get_unanswered_questions())
                }
                for name, tree in self.trees.items()
            }
        }


# =============================================================================
# AGI ULTRA: THOUGHT GRAPH - Global Graph Connecting All Trees
# =============================================================================

class ThoughtGraph:
    """
    AGI Ultra: Global graph that interconnects all module thought trees.
    
    Creates a unified knowledge graph where:
    - Each node from any module tree is a vertex
    - Edges represent reasoning dependencies, causal relationships, or semantic similarity
    - Enables cross-module reasoning and insight synthesis
    """
    
    def __init__(self, orchestrator: GlobalThoughtOrchestrator):
        self.orchestrator = orchestrator
        
        # Graph structure: adjacency list with edge metadata
        self.edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Node metadata cache for quick access
        self.node_cache: Dict[str, Dict[str, Any]] = {}
        
        # Edge types and their weights
        self.edge_weights = {
            'parent_child': 1.0,      # Hierarchical reasoning
            'cross_module': 0.8,      # Cross-module connection
            'semantic': 0.6,          # Semantic similarity
            'causal': 0.9,            # Causal relationship
            'temporal': 0.5,          # Temporal co-occurrence
            'contradiction': -0.7,    # Contradicting thoughts
        }
        
        # Statistics
        self.total_edges = 0
        self._lock = threading.Lock()
        
        logger.info("ThoughtGraph initialized")
    
    def build_graph(self) -> int:
        """
        Build the global thought graph from all module trees.
        
        Returns:
            Number of edges created
        """
        with self._lock:
            self.edges.clear()
            self.node_cache.clear()
            self.total_edges = 0
            
            # Process each tree
            for module_name, tree in self.orchestrator.trees.items():
                for node_id, node in tree.nodes.items():
                    # Cache node metadata
                    self.node_cache[node_id] = {
                        'module': module_name,
                        'question': node.question,
                        'answer': node.answer,
                        'confidence': node.confidence,
                        'timestamp': node.timestamp.isoformat() if node.timestamp else None,
                    }
                    
                    # Add parent-child edges
                    if node.parent_id and node.parent_id in tree.nodes:
                        self._add_edge(node.parent_id, node_id, 'parent_child')
                    
                    # Add explicit connections
                    for conn_id in node.connections:
                        self._add_edge(node_id, conn_id, 'semantic')
            
            # Add cross-module connections
            for key, connections in self.orchestrator.cross_module_connections.items():
                parts = key.split(':')
                if len(parts) == 2:
                    source_node = parts[1]
                    for conn in connections:
                        conn_parts = conn.split(':')
                        if len(conn_parts) == 2:
                            target_node = conn_parts[1]
                            self._add_edge(source_node, target_node, 'cross_module')
            
            logger.info(f"ThoughtGraph built: {len(self.node_cache)} nodes, {self.total_edges} edges")
            return self.total_edges
    
    def _add_edge(self, source: str, target: str, edge_type: str, metadata: Dict = None):
        """Add an edge to the graph."""
        edge = {
            'target': target,
            'type': edge_type,
            'weight': self.edge_weights.get(edge_type, 0.5),
            'metadata': metadata or {}
        }
        
        # Avoid duplicates
        existing = [e for e in self.edges[source] if e['target'] == target and e['type'] == edge_type]
        if not existing:
            self.edges[source].append(edge)
            self.total_edges += 1
    
    def find_path(self, start_node: str, end_node: str, max_depth: int = 10) -> List[str]:
        """
        Find reasoning path between two nodes using BFS.
        
        Returns:
            List of node IDs from start to end, or empty if no path
        """
        if start_node == end_node:
            return [start_node]
        
        visited = {start_node}
        queue = [(start_node, [start_node])]
        
        while queue and max_depth > 0:
            current, path = queue.pop(0)
            max_depth -= 1
            
            for edge in self.edges.get(current, []):
                next_node = edge['target']
                if next_node == end_node:
                    return path + [next_node]
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))
        
        return []
    
    def get_connected_thoughts(self, node_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """
        Get all thoughts connected to a node within a certain depth.
        
        Returns:
            List of connected node info with distance
        """
        connected = []
        visited = {node_id}
        current_level = [node_id]
        
        for d in range(depth):
            next_level = []
            for current in current_level:
                for edge in self.edges.get(current, []):
                    target = edge['target']
                    if target not in visited:
                        visited.add(target)
                        next_level.append(target)
                        connected.append({
                            'node_id': target,
                            'distance': d + 1,
                            'edge_type': edge['type'],
                            'weight': edge['weight'],
                            **self.node_cache.get(target, {})
                        })
            current_level = next_level
        
        return connected
    
    def find_contradictions(self) -> List[Tuple[str, str, str]]:
        """
        Find contradicting thoughts across the graph.
        
        Returns:
            List of (node1, node2, reason) tuples
        """
        contradictions = []
        
        # Simple heuristic: nodes with opposite answers in similar contexts
        for node_id, info in self.node_cache.items():
            if info.get('answer'):
                for other_id, other_info in self.node_cache.items():
                    if other_id != node_id and other_info.get('answer'):
                        # Check for BUY vs SELL contradictions
                        if ('BUY' in info['answer'] and 'SELL' in other_info['answer']) or \
                           ('SELL' in info['answer'] and 'BUY' in other_info['answer']):
                            # Check if questions are similar (simplified)
                            if self._questions_similar(info['question'], other_info['question']):
                                contradictions.append((
                                    node_id, 
                                    other_id, 
                                    f"Conflicting signals: {info['answer'][:20]} vs {other_info['answer'][:20]}"
                                ))
        
        return contradictions[:100]  # Limit results
    
    def _questions_similar(self, q1: str, q2: str) -> bool:
        """Check if two questions are semantically similar."""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return (intersection / union) > 0.5 if union > 0 else False


# =============================================================================
# AGI ULTRA: CONFIDENCE PROPAGATOR - Bayesian Confidence Spreading
# =============================================================================

class ConfidencePropagator:
    """
    AGI Ultra: Propagates confidence through the thought graph using Bayesian inference.
    
    When a node's confidence changes:
    - Parent nodes may increase/decrease confidence based on child evidence
    - Child nodes inherit some confidence from parents (prior)
    - Connected nodes influence each other based on edge weight
    """
    
    def __init__(self, graph: ThoughtGraph, orchestrator: GlobalThoughtOrchestrator):
        self.graph = graph
        self.orchestrator = orchestrator
        
        # Propagation parameters
        self.parent_influence = 0.3   # How much parent confidence affects children
        self.child_influence = 0.4    # How much children affect parent
        self.peer_influence = 0.2     # How much connected nodes affect each other
        self.decay_factor = 0.9       # Confidence decay per propagation step
        
        self._lock = threading.Lock()
        logger.info("ConfidencePropagator initialized")
    
    def propagate(self, source_node_id: str, new_confidence: float, max_steps: int = 5):
        """
        Propagate confidence change from a source node through the graph.
        
        Args:
            source_node_id: Node where confidence changed
            new_confidence: New confidence value (0-1)
            max_steps: Maximum propagation depth
        """
        with self._lock:
            # Find the source node
            source_module = None
            source_node = None
            
            for module_name, tree in self.orchestrator.trees.items():
                if source_node_id in tree.nodes:
                    source_module = module_name
                    source_node = tree.nodes[source_node_id]
                    break
            
            if not source_node:
                logger.warning(f"ConfidencePropagator: Node {source_node_id} not found")
                return
            
            # Update source confidence
            old_confidence = source_node.confidence
            source_node.confidence = new_confidence
            delta = new_confidence - old_confidence
            
            # Propagate to connected nodes
            self._propagate_recursive(source_node_id, delta, max_steps, visited=set())
    
    def _propagate_recursive(self, node_id: str, delta: float, steps_remaining: int, visited: Set[str]):
        """Recursively propagate confidence changes."""
        if steps_remaining <= 0 or node_id in visited or abs(delta) < 0.01:
            return
        
        visited.add(node_id)
        
        # Find node's tree
        for module_name, tree in self.orchestrator.trees.items():
            if node_id in tree.nodes:
                node = tree.nodes[node_id]
                
                # Propagate to parent
                if node.parent_id and node.parent_id in tree.nodes:
                    parent = tree.nodes[node.parent_id]
                    parent_delta = delta * self.child_influence * self.decay_factor
                    parent.confidence = max(0, min(1, parent.confidence + parent_delta))
                    self._propagate_recursive(node.parent_id, parent_delta, steps_remaining - 1, visited)
                
                # Propagate to children
                for child_id in node.children_ids:
                    if child_id in tree.nodes:
                        child = tree.nodes[child_id]
                        child_delta = delta * self.parent_influence * self.decay_factor
                        child.confidence = max(0, min(1, child.confidence + child_delta))
                        self._propagate_recursive(child_id, child_delta, steps_remaining - 1, visited)
                
                break
        
        # Propagate through graph edges
        for edge in self.graph.edges.get(node_id, []):
            target_id = edge['target']
            if target_id not in visited:
                # Find target node
                for mod_name, tree in self.orchestrator.trees.items():
                    if target_id in tree.nodes:
                        target = tree.nodes[target_id]
                        edge_delta = delta * self.peer_influence * edge['weight'] * self.decay_factor
                        target.confidence = max(0, min(1, target.confidence + edge_delta))
                        self._propagate_recursive(target_id, edge_delta, steps_remaining - 1, visited)
                        break
    
    def compute_posterior(self, node_id: str) -> float:
        """
        Compute Bayesian posterior confidence for a node based on its neighborhood.
        
        Returns:
            Updated confidence estimate
        """
        # Find node
        node = None
        for tree in self.orchestrator.trees.values():
            if node_id in tree.nodes:
                node = tree.nodes[node_id]
                break
        
        if not node:
            return 0.0
        
        prior = node.confidence
        
        # Collect evidence from connected nodes
        connected = self.graph.get_connected_thoughts(node_id, depth=2)
        
        if not connected:
            return prior
        
        # Weighted average of connected confidences
        total_weight = 0
        weighted_sum = 0
        
        for conn in connected:
            weight = conn['weight'] / conn['distance']  # Closer nodes have more influence
            confidence = conn.get('confidence', 0.5)
            weighted_sum += confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return prior
        
        likelihood = weighted_sum / total_weight
        
        # Simple Bayesian combination: prior * likelihood / normalization
        posterior = (prior * likelihood) / (prior * likelihood + (1 - prior) * (1 - likelihood) + 0.001)
        
        return posterior


# =============================================================================
# AGI ULTRA: TREE COMPRESSOR - Intelligent Compression of Old Trees
# =============================================================================

class TreeCompressor:
    """
    AGI Ultra: Compresses old/less important thought trees while preserving critical insights.
    
    Strategies:
    - Remove low-confidence leaf nodes
    - Merge similar nodes
    - Summarize subtrees into single compressed nodes
    - Archive old trees to disk
    """
    
    def __init__(self, orchestrator: GlobalThoughtOrchestrator, persistence_dir: str = "brain/thought_trees"):
        self.orchestrator = orchestrator
        self.persistence_dir = persistence_dir
        
        # Compression thresholds
        self.min_confidence_to_keep = 0.2   # Remove nodes below this
        self.max_nodes_per_tree = 10000     # Trigger compression above this
        self.similarity_threshold = 0.8     # Merge nodes above this similarity
        self.archive_age_days = 7           # Archive trees older than this
        
        os.makedirs(persistence_dir, exist_ok=True)
        
        self._lock = threading.Lock()
        logger.info(f"TreeCompressor initialized: persistence_dir={persistence_dir}")
    
    def compress_tree(self, module_name: str) -> Dict[str, int]:
        """
        Compress a single module's thought tree.
        
        Returns:
            Statistics about compression (nodes_removed, nodes_merged, etc.)
        """
        with self._lock:
            if module_name not in self.orchestrator.trees:
                return {'error': 'Tree not found'}
            
            tree = self.orchestrator.trees[module_name]
            original_count = len(tree.nodes)
            
            stats = {
                'original_nodes': original_count,
                'removed_low_confidence': 0,
                'merged_similar': 0,
                'final_nodes': 0
            }
            
            # Phase 1: Remove low-confidence leaf nodes
            nodes_to_remove = []
            for node_id, node in tree.nodes.items():
                # Is it a leaf node?
                if not node.children_ids:
                    # Low confidence and not root
                    if node.confidence < self.min_confidence_to_keep and node.parent_id:
                        nodes_to_remove.append(node_id)
            
            for node_id in nodes_to_remove:
                self._remove_node(tree, node_id)
                stats['removed_low_confidence'] += 1
            
            # Phase 2: Merge similar sibling nodes
            for node_id, node in list(tree.nodes.items()):
                if node_id not in tree.nodes:  # Already removed
                    continue
                    
                for sibling_id in node.connections[:]:
                    if sibling_id in tree.nodes:
                        sibling = tree.nodes[sibling_id]
                        if self._nodes_similar(node, sibling):
                            self._merge_nodes(tree, node_id, sibling_id)
                            stats['merged_similar'] += 1
            
            stats['final_nodes'] = len(tree.nodes)
            
            logger.info(f"Compressed {module_name}: {original_count} -> {len(tree.nodes)} nodes")
            
            return stats
    
    def compress_all(self) -> Dict[str, Dict[str, int]]:
        """Compress all trees that exceed the node limit."""
        results = {}
        
        for module_name, tree in list(self.orchestrator.trees.items()):
            if len(tree.nodes) > self.max_nodes_per_tree:
                results[module_name] = self.compress_tree(module_name)
        
        return results
    
    def _remove_node(self, tree: ThoughtTree, node_id: str):
        """Remove a node from the tree."""
        if node_id not in tree.nodes:
            return
        
        node = tree.nodes[node_id]
        
        # Remove from parent's children list
        if node.parent_id and node.parent_id in tree.nodes:
            parent = tree.nodes[node.parent_id]
            if node_id in parent.children_ids:
                parent.children_ids.remove(node_id)
        
        # Remove from root nodes if applicable
        if node_id in tree.root_nodes:
            tree.root_nodes.remove(node_id)
        
        # Reparent children to grandparent (or make them roots)
        for child_id in node.children_ids:
            if child_id in tree.nodes:
                tree.nodes[child_id].parent_id = node.parent_id
                if node.parent_id and node.parent_id in tree.nodes:
                    tree.nodes[node.parent_id].add_child(child_id)
                elif child_id not in tree.root_nodes:
                    tree.root_nodes.append(child_id)
        
        # Delete the node
        del tree.nodes[node_id]
    
    def _merge_nodes(self, tree: ThoughtTree, keep_id: str, merge_id: str):
        """Merge two nodes, keeping the first one."""
        if keep_id not in tree.nodes or merge_id not in tree.nodes:
            return
        
        keep_node = tree.nodes[keep_id]
        merge_node = tree.nodes[merge_id]
        
        # Combine confidence (weighted average)
        keep_node.confidence = (keep_node.confidence + merge_node.confidence) / 2
        
        # Combine children
        for child_id in merge_node.children_ids:
            if child_id in tree.nodes:
                tree.nodes[child_id].parent_id = keep_id
                keep_node.add_child(child_id)
        
        # Combine connections
        for conn_id in merge_node.connections:
            keep_node.add_connection(conn_id)
        
        # Append answer if exists
        if merge_node.answer and keep_node.answer:
            keep_node.answer = f"{keep_node.answer}; {merge_node.answer}"
        elif merge_node.answer:
            keep_node.answer = merge_node.answer
        
        # Remove merged node
        self._remove_node(tree, merge_id)
    
    def _nodes_similar(self, node1: ThoughtNode, node2: ThoughtNode) -> bool:
        """Check if two nodes are similar enough to merge."""
        words1 = set(node1.question.lower().split())
        words2 = set(node2.question.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return (intersection / union) >= self.similarity_threshold if union > 0 else False
    
    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------
    def save_tree(self, module_name: str) -> bool:
        """Save a tree to disk incrementally."""
        if module_name not in self.orchestrator.trees:
            return False
        
        tree = self.orchestrator.trees[module_name]
        filepath = os.path.join(self.persistence_dir, f"{module_name}.pkl")
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'module_name': tree.module_name,
                    'max_depth': tree.max_depth,
                    'nodes': {nid: self._serialize_node(n) for nid, n in tree.nodes.items()},
                    'root_nodes': tree.root_nodes
                }, f)
            logger.debug(f"Saved tree {module_name} to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save tree {module_name}: {e}")
            return False
    
    def load_tree(self, module_name: str) -> bool:
        """Load a tree from disk."""
        filepath = os.path.join(self.persistence_dir, f"{module_name}.pkl")
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            tree = ThoughtTree(data['module_name'], data['max_depth'])
            tree.root_nodes = data['root_nodes']
            
            for nid, node_data in data['nodes'].items():
                tree.nodes[nid] = self._deserialize_node(node_data)
            
            self.orchestrator.trees[module_name] = tree
            logger.debug(f"Loaded tree {module_name} from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load tree {module_name}: {e}")
            return False
    
    def save_all(self) -> int:
        """Save all trees to disk. Returns count of saved trees."""
        saved = 0
        for module_name in self.orchestrator.trees:
            if self.save_tree(module_name):
                saved += 1
        return saved
    
    def _serialize_node(self, node: ThoughtNode) -> Dict[str, Any]:
        """Serialize a node for persistence."""
        return {
            'node_id': node.node_id,
            'module_name': node.module_name,
            'question': node.question,
            'answer': node.answer,
            'context': node.context,
            'confidence': node.confidence,
            'timestamp': node.timestamp.isoformat() if node.timestamp else None,
            'parent_id': node.parent_id,
            'children_ids': node.children_ids,
            'connections': node.connections,
            'metadata': node.metadata
        }
    
    def _deserialize_node(self, data: Dict[str, Any]) -> ThoughtNode:
        """Deserialize a node from persistence."""
        return ThoughtNode(
            node_id=data['node_id'],
            module_name=data['module_name'],
            question=data['question'],
            answer=data.get('answer'),
            context=data.get('context', {}),
            confidence=data.get('confidence', 0.0),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else datetime.now(),
            parent_id=data.get('parent_id'),
            children_ids=data.get('children_ids', []),
            connections=data.get('connections', []),
            metadata=data.get('metadata', {})
        )
