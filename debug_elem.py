import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from ShellFemSolver.shell_solver import (
    _K_bending_tria3, _K_shear_dsg3, _B_membrane_tria3, compute_tria3_local
)

# Single element test
x2d = jnp.array([0., 40., 0.])
y2d = jnp.array([0., 0., 40.])
area = 800.0
E, t, nu, rho = 210000.0, 1.0, 0.3, 7.85e-9

K_b = _K_bending_tria3(E, t, nu, area, x2d, y2d)
K_s = _K_shear_dsg3(E, t, nu, area, x2d, y2d)

print('Max K_bend:', float(jnp.max(jnp.abs(K_b))))
print('Max K_shear:', float(jnp.max(jnp.abs(K_s))))
print('Ratio shear/bend:', float(jnp.max(jnp.abs(K_s))) / float(jnp.max(jnp.abs(K_b))))

# Theoretical D for reference
D = E*t**3/(12*(1-nu**2))
print(f'D = {D:.4f}')

# Shear modulus * t
G = E/(2*(1+nu))
print(f'G*t = {G*t:.4f}')
print(f'D/(G*t) = {D/(G*t):.6f}  (should be ~ (t/L)^2 for thin plate)')
