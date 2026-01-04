
import os
import sys
import subprocess
import ctypes

def compile_cpp():
    print("[BUILD] Starting C++ Compilation...")
    
    # 1. Detect Compiler
    # Try MinGW first (common on lightweight setups)
    compiler = "g++"
    
    # Check if g++ exists
    try:
        subprocess.run(["g++", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("[WARN] g++ not found. Trying cl.exe (MSVC)...")
        compiler = "cl" # TODO: Implement MSVC logic if needed, usually requires vcvarsall.bat
        
    src = "cpp_core/mcts.cpp"
    out = "cpp_core/mcts_core.dll"
    
    if compiler == "g++":
        # Compile shared library
        cmd = [
            "g++", 
            "-shared", 
            "-o", out, 
            src, 
            "-O3", # Optimize for speed
            "-std=c++17",
            "-static-libgcc", 
            "-static-libstdc++" # Avoid dependency hell
        ]
        
        print(f"[CMD] {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"[SUCCESS] Compiled {out}")
                return True
            else:
                print(f"[ERROR] Compilation Failed:\n{res.stderr}")
                return False
        except Exception as e:
            print(f"[ERROR] Build Exception: {e}")
            return False
            
    else:
        print("[ERROR] No suitable compiler found. Install MinGW-w64.")
        return False

if __name__ == "__main__":
    compile_cpp()
