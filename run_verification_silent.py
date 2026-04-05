import subprocess
import time
import os

def run_script():
    cmd = ["python", "c:\\Users\\GOODMAN\\code_sheet\\main_shell_verification.py"]
    # Run with current environment
    env = os.environ.copy()
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd="c:\\Users\\GOODMAN\\code_sheet", env=env)
    
    try:
        # 1. Load from cache? [y/N]
        time.sleep(5)
        process.stdin.write("y\n")
        process.stdin.flush()
        
        # 2. View results (Select choice: 0)
        time.sleep(3)
        process.stdin.write("0\n")
        process.stdin.flush()
        
        # 3. Final choice? (Iteration 0 best, it will ask for input again if needed)
        time.sleep(15) 
        process.stdin.write("0\n")
        process.stdin.flush()
        
        # Capture remaining output
        stdout, stderr = process.communicate(timeout=600)
        print(stdout)
        print(stderr)
    except Exception as e:
        process.kill()
        print(f"Error or Timeout: {e}")
        outs, errs = process.communicate()
        print(outs)
        print(errs)

if __name__ == "__main__":
    run_script()
