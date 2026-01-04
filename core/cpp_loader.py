
import ctypes
import os
import logging
import sys

logger = logging.getLogger("CPPLoader")

def _add_mingw_to_path():
    """
    Adds MinGW bin directory to DLL search path.
    Crucial for finding dependencies like libstdc++-6.dll.
    """
    # Common MSYS2 Locations
    mingw_paths = [
        r"D:\msys64\ucrt64\bin",
        r"C:\msys64\ucrt64\bin",
        r"D:\msys64\mingw64\bin",
        r"C:\msys64\mingw64\bin",
        r"C:\ProgramData\chocolatey\bin"
    ]
    
    found = False
    for p in mingw_paths:
        if os.path.exists(p):
            try:
                # Python 3.8+ specific
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(p)
                else:
                    os.environ['PATH'] = p + os.pathsep + os.environ['PATH']
                # logger.debug(f"Added DLL Directory: {p}")
                found = True
                break
            except Exception as e:
                logger.warning(f"Failed to add DLL dir {p}: {e}")
                
    if not found:
        # logger.warning("MinGW bin directory not found! C++ DLLs might fail to load.")
        pass

# Run once on import
_add_mingw_to_path()

def load_dll(dll_name: str) -> ctypes.CDLL:
    """
    Loads a DLL from the cpp_core directory.
    dll_name: e.g. "mcts_core.dll"
    """
    # Construct absolute path to cpp_core
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dll_path = os.path.join(base_dir, "cpp_core", dll_name)
    
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"DLL not found at: {dll_path}")
        
    # Load Library
    # The dependencies should now be resolvable thanks to _add_mingw_to_path
    lib = ctypes.CDLL(dll_path)
    return lib
