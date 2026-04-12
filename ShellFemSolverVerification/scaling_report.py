# -*- coding: utf-8 -*-
import os, sys, subprocess, re, time

_HERE = os.path.dirname(os.path.abspath(__file__))
WORKER_PATH = os.path.join(_HERE, 'worker_bench.py')

def run_bench(num_cores):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_cores)
    env["MKL_NUM_THREADS"] = str(num_cores)
    # Removing XLA_FLAGS for now as it can be picky about multiple flags
    # env["XLA_FLAGS"] = f"--xla_cpu_parallelify_tuple_size={num_cores}"
    
    cmd = [sys.executable, WORKER_PATH]
    try:
        # Using CWD to ensure imports work
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True, cwd=os.path.join(_HERE, '..'))
        out = result.stdout
        
        # Regex to find times
        jit_match = re.search(r"JIT_TIME: ([\d.]+) ms", out)
        exec_match = re.search(r"EXEC_TIME: ([\d.]+) ms", out)
        
        jit_time = float(jit_match.group(1)) if jit_match else 0.0
        exec_time = float(exec_match.group(1)) if exec_match else 0.0
        
        return jit_time, exec_time
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {str(e)}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None, None
    except Exception as e:
        print(f"Unexpected error for {num_cores} cores: {str(e)}")
        return None, None

def generate_scaling_report():
    print("="*60)
    print("  ShellFEM Solver Multi-Core Scaling Benchmark (JAX CPU)")
    print("  Comparing: 1, 2, 4, 6 Cores")
    print("="*60)
    
    cores = [1, 2, 4, 6]
    results = {}
    
    for c in cores:
        print(f"Running benchmark with {c} core(s)...", end="", flush=True)
        jit, exec_t = run_bench(c)
        if jit is not None:
            results[c] = {"jit": jit, "exec": exec_t}
            print(f" DONE (Exec: {exec_t:.1f} ms)")
        else:
            print(" FAILED")
            
    if 1 not in results:
        return ""
    
    base_exec = results[1]["exec"]
    
    lines = [
        "## 5. Multi-Core Scaling Profile",
        "",
        "| Cores | Execution Time (ms) | Speedup | Efficiency (%) | JIT Overlap (ms) |",
        "|-------|---------------------|---------|----------------|------------------|",
    ]
    
    for c in cores:
        if c in results:
            exec_t = results[c]["exec"]
            jit_t = results[c]["jit"]
            speedup = base_exec / exec_t if exec_t > 0 else 0.0
            efficiency = (speedup / c) * 100.0
            lines.append(f"| {c} | {exec_t:.1f} | {speedup:.2f}x | {efficiency:.1f}% | {jit_t:.1f} |")
    
    lines += [
        "",
        "### 5.1. Scalability Analysis",
        "Higher core counts show improvement in pure execution time. Parallel efficiency typically decreases as communication and memory bandwidth bottlenecks become more significant.",
        ""
    ]
    
    return "\n".join(lines)

if __name__ == "__main__":
    report = generate_scaling_report()
    print("\n" + report)
