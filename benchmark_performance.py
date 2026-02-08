import time
import jax
import jax.numpy as jnp
from solver import PlateFEM

# Enable JAX precision
jax.config.update("jax_enable_x64", True)

def benchmark_solver(nx=20, ny=10):
    print(f"Benchmarking (nx={nx}, ny={ny})...")
    
    # 1. Setup
    start_setup = time.time()
    fem = PlateFEM(Lx=200.0, Ly=100.0, nx=nx, ny=ny)
    num_nodes = (nx+1)*(ny+1)
    
    params = {
        't': jnp.full(num_nodes, 1.0),
        'rho': jnp.full(num_nodes, 7.85e-9),
        'E': jnp.full(num_nodes, 210000.0),
        'z': jnp.full(num_nodes, 0.0)
    }
    
    # Simple static load
    F = jnp.zeros(fem.total_dof)
    F = F.at[2::6].set(10.0) # Downward force on all nodes
    
    fixed_dof = jnp.arange(0, fem.total_dof, 6) # Fix u (example)
    # Actually fix left edge completely
    # Left edge nodes: 0, 1, ..., ny
    left_nodes = jnp.arange(ny+1)
    fixed_dofs = []
    for n in left_nodes:
        fixed_dofs.extend([n*6+i for i in range(6)])
    fixed_dofs = jnp.array(fixed_dofs)
    
    print(f"Setup Time: {time.time() - start_setup:.4f}s")
    
    # Pre-compute free DOFs indices outside JIT to avoid dynamic shape error
    mask = jnp.ones(fem.total_dof, dtype=bool)
    mask = mask.at[fixed_dofs].set(False)
    free_dofs = jnp.where(mask)[0]
    
    # 2. Assemble & Solve (Function to JIT)
    @jax.jit
    def run_step(params):
        K, M = fem.assemble(params)
        
        # Slicing with pre-computed indices is JIT-compatible
        K_ff = K[jnp.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        
        # Solve
        u_f = fem.solve_static(K_ff, F_f)
        return jnp.sum(u_f)
    
    # 3. Compilation Run
    print("Compiling (First Run)...")
    start_compile = time.time()
    # Block until ready
    res = run_step(params).block_until_ready()
    end_compile = time.time()
    compile_time = end_compile - start_compile
    print(f"Compilation Time: {compile_time:.4f}s")
    
    # 4. Execution Run (Average)
    runs = 100
    print(f"Running {runs} iterations...")
    start_exec = time.time()
    for _ in range(runs):
        _ = run_step(params).block_until_ready()
    end_exec = time.time()
    
    avg_time = (end_exec - start_exec) / runs
    print(f"Average Execution Time: {avg_time*1000:.2f}ms per iteration")
    
    return compile_time, avg_time

if __name__ == "__main__":
    benchmark_solver(20, 10) # Small (231 nodes)
    print("-" * 30)
    benchmark_solver(50, 20) # Medium (1071 nodes)
    print("-" * 30)
    benchmark_solver(80, 40) # Large (3321 nodes, ~20k DOFs)
