"""
AGI Fase 2: Self-Modification System

Sistema de Auto-Modificação Segura:
- Modificação de Código em Sandbox
- Adição de Módulos
- Refatoração Autônoma
- Expansão de Capacidades
"""

import logging
import time
import os
import ast
import copy
import hashlib
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("SelfModification")


class ModificationType(Enum):
    PARAMETER = "parameter"
    FUNCTION = "function"
    MODULE = "module"
    REFACTOR = "refactor"
    EXPANSION = "expansion"


class ModificationStatus(Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    VALIDATED = "validated"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


@dataclass
class Modification:
    """A proposed modification to the system."""
    mod_id: str
    mod_type: ModificationType
    target: str  # File or module to modify
    description: str
    original_content: Optional[str] = None
    new_content: Optional[str] = None
    status: ModificationStatus = ModificationStatus.PROPOSED
    
    # Testing
    test_results: List[Dict] = field(default_factory=list)
    safety_score: float = 0.0
    
    # Meta
    created_at: float = field(default_factory=time.time)
    applied_at: Optional[float] = None
    rolled_back_at: Optional[float] = None


@dataclass
class Checkpoint:
    """A system state checkpoint for rollback."""
    checkpoint_id: str
    timestamp: float
    files: Dict[str, str]  # filename -> content
    metadata: Dict[str, Any]


class SandboxEnvironment:
    """
    Isolated sandbox for testing modifications.
    
    Prevents modifications from affecting live system.
    """
    
    def __init__(self, sandbox_dir: str = "sandbox"):
        self.sandbox_dir = sandbox_dir
        self.active = False
        
        logger.info(f"SandboxEnvironment initialized at {sandbox_dir}")
    
    def enter(self):
        """Enter sandbox mode."""
        self.active = True
        os.makedirs(self.sandbox_dir, exist_ok=True)
        logger.info("Entered sandbox mode")
    
    def exit(self):
        """Exit sandbox mode."""
        self.active = False
        logger.info("Exited sandbox mode")
    
    def execute_safely(
        self,
        code: str,
        context: Optional[Dict] = None
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute code safely in sandbox.
        
        Returns: (success, result, error)
        """
        if not self.active:
            self.enter()
        
        try:
            # Parse code for safety
            tree = ast.parse(code)
            
            # Check for dangerous operations
            dangerous_nodes = self._find_dangerous_nodes(tree)
            if dangerous_nodes:
                return False, None, f"Dangerous code detected: {dangerous_nodes}"
            
            # Execute in isolated namespace
            sandbox_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'abs': abs,
                    'round': round,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                }
            }
            
            if context:
                sandbox_globals.update(context)
            
            sandbox_locals = {}
            exec(code, sandbox_globals, sandbox_locals)
            
            return True, sandbox_locals, None
            
        except SyntaxError as e:
            return False, None, f"Syntax error: {e}"
        except Exception as e:
            return False, None, f"Execution error: {e}"
    
    def _find_dangerous_nodes(self, tree: ast.AST) -> List[str]:
        """Find potentially dangerous AST nodes."""
        dangerous = []
        
        for node in ast.walk(tree):
            # Check for imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'sys', 'subprocess', 'shutil']:
                        dangerous.append(f"import {alias.name}")
            
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile', 'open']:
                        dangerous.append(f"call {node.func.id}")
        
        return dangerous


class CodeModifier:
    """
    Modifies code with safety checks.
    
    Ensures modifications are valid and safe.
    """
    
    def __init__(self, sandbox: SandboxEnvironment):
        self.sandbox = sandbox
        self.modifications: Dict[str, Modification] = {}
        self._mod_counter = 0
        
        logger.info("CodeModifier initialized")
    
    def propose_modification(
        self,
        target: str,
        mod_type: ModificationType,
        description: str,
        new_content: str
    ) -> Modification:
        """Propose a new modification."""
        self._mod_counter += 1
        mod_id = f"mod_{self._mod_counter}"
        
        # Read original content if file exists
        original = None
        if os.path.exists(target):
            try:
                with open(target, 'r') as f:
                    original = f.read()
            except Exception:
                pass
        
        mod = Modification(
            mod_id=mod_id,
            mod_type=mod_type,
            target=target,
            description=description,
            original_content=original,
            new_content=new_content
        )
        
        self.modifications[mod_id] = mod
        logger.info(f"Modification proposed: {mod_id} - {description}")
        return mod
    
    def validate_modification(self, mod: Modification) -> Tuple[bool, str]:
        """Validate a modification is syntactically correct."""
        if not mod.new_content:
            return False, "No new content"
        
        # Try to parse as Python
        if mod.target.endswith('.py'):
            try:
                ast.parse(mod.new_content)
                return True, "Syntax valid"
            except SyntaxError as e:
                return False, f"Syntax error: {e}"
        
        return True, "Non-Python file, skipping validation"
    
    def test_modification(self, mod: Modification) -> float:
        """Test modification in sandbox. Returns safety score."""
        mod.status = ModificationStatus.TESTING
        
        scores = []
        
        # Syntax validation
        valid, msg = self.validate_modification(mod)
        scores.append(1.0 if valid else 0.0)
        mod.test_results.append({'test': 'syntax', 'passed': valid, 'msg': msg})
        
        if not valid:
            mod.safety_score = 0.0
            mod.status = ModificationStatus.REJECTED
            return 0.0
        
        # Sandbox execution test
        if mod.new_content and mod.target.endswith('.py'):
            success, result, error = self.sandbox.execute_safely(mod.new_content)
            scores.append(1.0 if success else 0.5)
            mod.test_results.append({
                'test': 'sandbox',
                'passed': success,
                'error': error
            })
        
        # Calculate safety score
        mod.safety_score = sum(scores) / len(scores) if scores else 0.0
        
        if mod.safety_score >= 0.8:
            mod.status = ModificationStatus.VALIDATED
        else:
            mod.status = ModificationStatus.REJECTED
        
        return mod.safety_score


class ModuleAdder:
    """
    Adds new modules to the system.
    
    Creates new functionality when needed.
    """
    
    def __init__(self, base_path: str = "core/agi"):
        self.base_path = base_path
        self.added_modules: List[str] = []
        
        logger.info("ModuleAdder initialized")
    
    def create_module_template(
        self,
        module_name: str,
        class_name: str,
        methods: List[Dict[str, str]]
    ) -> str:
        """Generate module template code."""
        method_code = []
        for method in methods:
            method_code.append(f'''
    def {method['name']}(self{', ' + method.get('args', '') if method.get('args') else ''}):
        """{method.get('doc', 'Method description.')}"""
        {method.get('body', 'pass')}
''')
        
        template = f'''"""
Auto-generated module: {module_name}

Created by AGI Self-Modification System.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("{module_name}")


class {class_name}:
    """Auto-generated class."""
    
    def __init__(self):
        logger.info("{class_name} initialized")
{''.join(method_code)}

# Module initialization
_instance: Optional[{class_name}] = None

def get_instance() -> {class_name}:
    global _instance
    if _instance is None:
        _instance = {class_name}()
    return _instance
'''
        return template
    
    def add_module(
        self,
        module_name: str,
        content: str,
        dry_run: bool = True
    ) -> Tuple[bool, str]:
        """Add a new module."""
        filepath = os.path.join(self.base_path, f"{module_name}.py")
        
        if dry_run:
            logger.info(f"DRY RUN: Would create {filepath}")
            return True, f"Would create {filepath}"
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(content)
            
            self.added_modules.append(filepath)
            logger.info(f"Module created: {filepath}")
            return True, f"Created {filepath}"
        except Exception as e:
            return False, f"Failed to create module: {e}"


class AutonomousRefactorer:
    """
    Refactors code for improved efficiency.
    
    Identifies and applies common refactoring patterns.
    """
    
    def __init__(self):
        self.refactorings: List[Dict] = []
        
        logger.info("AutonomousRefactorer initialized")
    
    def analyze_code(self, code: str) -> List[Dict[str, Any]]:
        """Analyze code for refactoring opportunities."""
        opportunities = []
        
        try:
            tree = ast.parse(code)
            
            # Check for long functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                    if lines > 50:
                        opportunities.append({
                            'type': 'split_function',
                            'target': node.name,
                            'reason': f'Function too long ({lines} lines)',
                            'priority': 0.7
                        })
                
                # Check for duplicate code patterns
                if isinstance(node, ast.For):
                    if isinstance(node.body[0] if node.body else None, ast.Expr):
                        opportunities.append({
                            'type': 'list_comprehension',
                            'target': 'for_loop',
                            'reason': 'Could be list comprehension',
                            'priority': 0.5
                        })
        
        except SyntaxError:
            pass
        
        return opportunities
    
    def suggest_refactoring(
        self,
        code: str,
        opportunity: Dict[str, Any]
    ) -> Optional[str]:
        """Suggest refactored code."""
        # This would contain actual refactoring logic
        # For now, return placeholder
        return None


class CapabilityExpander:
    """
    Expands system capabilities based on need.
    
    Identifies gaps and proposes expansions.
    """
    
    def __init__(self):
        self.expansions: List[Dict] = []
        
        logger.info("CapabilityExpander initialized")
    
    def identify_gap(
        self,
        required_capability: str,
        current_capabilities: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Identify capability gap."""
        if required_capability not in current_capabilities:
            return {
                'gap': required_capability,
                'severity': 0.7,
                'suggested_implementation': f"Implement {required_capability} module"
            }
        return None
    
    def propose_expansion(
        self,
        gap: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Propose capability expansion."""
        expansion = {
            'capability': gap['gap'],
            'type': 'new_module',
            'description': f"Add capability: {gap['gap']}",
            'estimated_complexity': 0.5,
            'created_at': time.time()
        }
        
        self.expansions.append(expansion)
        return expansion


class CheckpointManager:
    """
    Manages system checkpoints for rollback.
    
    Enables safe experimentation.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoints: Dict[str, Checkpoint] = {}
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info("CheckpointManager initialized")
    
    def create_checkpoint(
        self,
        files: List[str],
        metadata: Optional[Dict] = None
    ) -> Checkpoint:
        """Create a checkpoint of current state."""
        checkpoint_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
        
        file_contents = {}
        for filepath in files:
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        file_contents[filepath] = f.read()
                except Exception:
                    pass
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            files=file_contents,
            metadata=metadata or {}
        )
        
        self.checkpoints[checkpoint_id] = checkpoint
        logger.info(f"Checkpoint created: {checkpoint_id}")
        return checkpoint
    
    def rollback(self, checkpoint_id: str) -> Tuple[bool, str]:
        """Rollback to a checkpoint."""
        if checkpoint_id not in self.checkpoints:
            return False, "Checkpoint not found"
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        restored = 0
        for filepath, content in checkpoint.files.items():
            try:
                with open(filepath, 'w') as f:
                    f.write(content)
                restored += 1
            except Exception as e:
                logger.error(f"Failed to restore {filepath}: {e}")
        
        logger.info(f"Rolled back to {checkpoint_id}, restored {restored} files")
        return True, f"Restored {restored} files"


class SelfModificationSystem:
    """
    Main Self-Modification System.
    
    Orchestrates safe self-modification with:
    - Sandbox testing
    - Validation
    - Rollback capability
    """
    
    def __init__(self):
        self.sandbox = SandboxEnvironment()
        self.modifier = CodeModifier(self.sandbox)
        self.module_adder = ModuleAdder()
        self.refactorer = AutonomousRefactorer()
        self.expander = CapabilityExpander()
        self.checkpoint_mgr = CheckpointManager()
        
        logger.info("SelfModificationSystem initialized")
    
    def safe_modify(
        self,
        target: str,
        new_content: str,
        description: str,
        auto_apply: bool = False
    ) -> Tuple[bool, Modification]:
        """
        Safely modify a file with full validation.
        
        1. Create checkpoint
        2. Propose modification
        3. Validate in sandbox
        4. Apply if safe
        """
        # 1. Checkpoint
        checkpoint = self.checkpoint_mgr.create_checkpoint([target])
        
        # 2. Propose
        mod = self.modifier.propose_modification(
            target=target,
            mod_type=ModificationType.FUNCTION,
            description=description,
            new_content=new_content
        )
        
        # 3. Validate
        safety_score = self.modifier.test_modification(mod)
        
        if safety_score < 0.8:
            return False, mod
        
        # 4. Apply if auto
        if auto_apply:
            try:
                with open(target, 'w') as f:
                    f.write(new_content)
                mod.status = ModificationStatus.APPLIED
                mod.applied_at = time.time()
                logger.info(f"Modification applied: {mod.mod_id}")
            except Exception as e:
                # Rollback
                self.checkpoint_mgr.rollback(checkpoint.checkpoint_id)
                mod.status = ModificationStatus.ROLLED_BACK
                mod.rolled_back_at = time.time()
                return False, mod
        
        return True, mod
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'modifications_proposed': len(self.modifier.modifications),
            'modifications_applied': len([
                m for m in self.modifier.modifications.values()
                if m.status == ModificationStatus.APPLIED
            ]),
            'modules_added': len(self.module_adder.added_modules),
            'checkpoints': len(self.checkpoint_mgr.checkpoints),
            'expansions_proposed': len(self.expander.expansions)
        }
