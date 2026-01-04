
import os
import sys
import subprocess
import ctypes
import shutil

def find_mingw():
    """Tries to locate MinGW G++ in common locations."""
    
    # 1. Check PATH
    if shutil.which("g++"):
        return "g++"
        
    # 2. Check Common MSYS2 Paths
    potential_paths = [
        r"D:\msys64\ucrt64\bin",
        r"C:\msys64\ucrt64\bin",
        r"D:\msys64\mingw64\bin",
        r"C:\msys64\mingw64\bin",
        r"C:\ProgramData\chocolatey\bin"
    ]
    
    for p in potential_paths:
        gpp_path = os.path.join(p, "g++.exe")
        if os.path.exists(gpp_path):
            print(f"[SETUP] Found g++ at: {gpp_path}")
            # Add to PATH for this session
            os.environ["PATH"] += os.pathsep + p
            return "g++"
            
    return None

def compile_cpp():
    print("[BUILD] Starting C++ Compilation...")
    
    compiler = find_mingw()
    
    if not compiler:
        print("[ERROR] g++ not found in PATH or standard MSYS2 folders (D:\\msys64...).")
        print("Please ensure 'D:\\msys64\\ucrt64\\bin' is in your System PATH.")
        return False
        
    targets = [
        {"src": "cpp_core/mcts.cpp", "out": "cpp_core/mcts_core.dll"},
        {"src": "cpp_core/physics.cpp", "out": "cpp_core/physics_core.dll"},
        {"src": "cpp_core/hdc.cpp", "out": "cpp_core/hdc_core.dll"}
    ]
    
    all_success = True
    
    for target in targets:
        src = target["src"]
        out = target["out"]
        
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
                print(f"[SUCESSO] Compiled {out}")
            else:
                print(f"[ERROR] Compilation Failed for {out}:\n{res.stderr}")
                all_success = False
        except Exception as e:
            print(f"[ERROR] Build Exception: {e}")
            all_success = False

    return all_success

if __name__ == "__main__":
    compile_cpp()
