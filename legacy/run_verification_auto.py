import subprocess
import time

def run_script():
    cmd = ["python", "c:\\Users\\GOODMAN\\code_sheet\\main_shell_verification.py"]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd="c:\\Users\\GOODMAN\\code_sheet")
    
    # 1. Load from cache? [y/N]
    time.sleep(5)
    process.stdin.write("y\n")
    process.stdin.flush()
    
    # 2. Pattern check choice [0-2]
    time.sleep(2)
    process.stdin.write("0\n")
    process.stdin.flush()
    
    # 3. Optimization choice [0-...]
    time.sleep(10) # Wait for GT loading/gen
    process.stdin.write("0\n")
    process.stdin.flush()
    
    # Wait for completion or timeout
    try:
        stdout, stderr = process.communicate(timeout=300)
        print(stdout)
        print(stderr)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        print("Timeout reached.")
        print(stdout)
        print(stderr)

if __name__ == "__main__":
    run_script()
