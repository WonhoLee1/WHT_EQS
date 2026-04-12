# -*- coding: utf-8 -*-
import datetime
import tqdm
import jax.debug
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from ShellFemSolver.shell_solver import ShellFEM
from ShellFemSolver.mesh_utils import generate_rect_mesh_quads, generate_tray_mesh_quads
import WHT_EQS_mesh as mesh_utils
import h5py
import os
import subprocess
import pyvista as pv

class StructuralResult:
    """Container for FEM analysis results with interpolation support."""
    def __init__(self, nodal_results, node_coords, elements):
        self.results = nodal_results
        self.nodes = node_coords
        
        # Elements can be a single array or a list of [Quads, Trias]
        if isinstance(elements, list):
            self.elements = []
            for e in elements:
                if len(e) > 0:
                    self.elements.extend([list(row) for row in e])
        else:
            self.elements = [list(row) for row in elements]
            
        self._interpolators = {}

    def get_nodal_result(self, field_name):
        return self.results.get(field_name)

    def get_element_result(self, field_name):
        # Result averaging is usually done in ShellFEM.compute_field_results
        # Here we provide access to those fields.
        return self.results.get(f"{field_name}_el")

    def get_value_at_point(self, field_name, x, y, z):
        """
        Interpolates the result at a given 3D coordinate using surface projection logic.
        """
        data = self.results.get(field_name)
        if data is None: return None
        
        # 1. Find the nearest node (Global search)
        query_pt = np.array([x, y, z])
        dist_sq = np.sum((self.nodes - query_pt)**2, axis=1)
        nearest_idx = np.argmin(dist_sq)
        
        # 2. Optimized Projection: If the point is very close to a node, return its value
        if dist_sq[nearest_idx] < 1e-4:
            return float(data[nearest_idx])
            
        # 3. For true interpolation, we use a local neighborhood or global linear interpolator
        # [PROJECTION LOGIC] We use the provided data directly for the shell surface.
        if field_name not in self._interpolators:
            # We use LinearNDInterpolator which handles the 'closest facet' logic internally 
            # for points near the convex hull of the nodes.
            self._interpolators[field_name] = LinearNDInterpolator(self.nodes, data)
        
        val = self._interpolators[field_name](x, y, z)
        
        # 4. Fallback: If outside (NaN), return the nearest node value
        if np.isnan(val):
            return float(data[nearest_idx])
            
        return float(val)

    def _get_pv_grid(self, scale=1.0):
        """Helper to create PyVista UnstructuredGrid for mixed meshes."""
        cells = []
        cell_types = []
        for e in self.elements:
            cells.append(len(e))
            cells.extend(e)
            cell_types.append(9 if len(e)==4 else 5)
            
        grid = pv.UnstructuredGrid(cells, cell_types, self.nodes)
        
        # Apply deformation if present
        u = self.results.get('u_full')
        if u is not None:
            # Ensure u is a numpy array and do safe point assignment
            u_vec = np.asarray(u).reshape(-1, 6)[:, :3]
            grid.points = np.array(grid.points) + u_vec * float(scale)
            
        return grid

    def show(self, field_name='stress_vm', scale=20.0):
        """Quick interactive 3D visualization using PyVista."""
        grid = self._get_pv_grid(scale=scale)
        grid.point_data[field_name] = self.results.get(field_name, np.zeros(len(self.nodes)))
            
        p = pv.Plotter(title=f"FEM Result: {field_name}")
        p.set_background("white")
        p.add_mesh(grid, scalars=field_name, cmap="coolwarm", show_edges=True)
        p.add_scalar_bar(title=field_name)
        p.show()

    def save_vtu(self, filepath):
        """Saves the result to a standard VTK XML file (.vtu)."""
        grid = self._get_pv_grid(scale=0.0) # Undeformed mesh for standard vtu
        for field, data in self.results.items():
            if not field.endswith("_el") and len(data) == len(self.nodes):
                grid.point_data[field] = data
        grid.save(filepath)
        return self

    def save_vtkhdf(self, filepath, steps_dict=None):
        """
        High-performance binary HDF5 export for ParaView (VTKHDF 1.1 Spec).
        Supports both single-step static and multi-step temporal (Modal) data.
        """
        num_points = self.nodes.shape[0]
        num_cells = len(self.elements)
        
        # Prepare Connectivity, Offsets, and Types
        conn = []
        offsets = [0]
        types = []
        for elem in self.elements:
            conn.extend(elem)
            offsets.append(len(conn))
            types.append(9 if len(elem) == 4 else 5) # 9: Quad, 5: Triangle
        
        num_conn_ids = len(conn)
        
        # --- Robust Filename Handling ---
        actual_path = filepath
        counter = 1
        base, ext = os.path.splitext(filepath)
        while True:
            try:
                f = h5py.File(actual_path, 'w')
                f.close()
                break
            except OSError:
                actual_path = f"{base}_{counter}{ext}"
                counter += 1
                if counter > 20: break
        
        with h5py.File(actual_path, 'w') as f:
            vtkhdf = f.create_group("VTKHDF")
            
            # 1. Identity
            vtkhdf.attrs.create("Version", [1, 1], dtype='i4') # Upgrade to 1.1 for temporal
            vtkhdf.attrs.create("Type", np.bytes_("UnstructuredGrid"))
            vtkhdf.create_dataset("Version", data=np.array([1, 1], dtype=np.int32))
            
            dt_str = h5py.string_dtype(encoding='ascii', length=32)
            ds_type = vtkhdf.create_dataset("Type", shape=(), dtype=dt_str)
            ds_type[()] = "UnstructuredGrid"
            
            # 2. Global Sizes
            vtkhdf.create_dataset("NumberOfPoints", data=np.array([num_points], dtype=np.int64))
            vtkhdf.create_dataset("NumberOfCells", data=np.array([num_cells], dtype=np.int64))
            vtkhdf.create_dataset("NumberOfConnectivityIds", data=np.array([num_conn_ids], dtype=np.int64))
            
            # 3. Mesh Data (Static for now)
            vtkhdf.create_dataset("Points", data=self.nodes.astype(np.float64))
            vtkhdf.create_dataset("Connectivity", data=np.array(conn, dtype=np.int64))
            vtkhdf.create_dataset("Offsets", data=np.array(offsets, dtype=np.int64))
            vtkhdf.create_dataset("Types", data=np.array(types, dtype=np.uint8))
            
            # 4. Temporal Steps
            if steps_dict:
                steps_group = vtkhdf.create_group("Steps")
                t_values = np.array(steps_dict['values'], dtype=np.float64)
                n_steps = len(t_values)
                steps_group.attrs.create("NSteps", n_steps, dtype='i4')
                steps_group.create_dataset("Values", data=t_values)
                
                # Geometry is static, so offsets repeat the whole mesh per step
                # For UnstructuredGrid, PartOffsets maps to the flattened PointData
                steps_group.create_dataset("PartOffsets", data=np.arange(n_steps, dtype=np.int64) * num_points)
                steps_group.create_dataset("NumberOfParts", data=np.ones(n_steps, dtype=np.int64))

            # 5. PointData (Multi-step if steps_dict present)
            pd = vtkhdf.create_group("PointData")
            
            if steps_dict:
                # Concatenate data across steps
                for field, step_data_list in steps_dict['point_data'].items():
                    # Each item in list is (num_points,) or (num_points, 3)
                    flat_data = []
                    is_vector = False
                    for d in step_data_list:
                        arr = np.asarray(d)
                        if arr.ndim == 1 and len(arr) == 6*num_points: # 6-DOF to 3-DOF
                            arr = arr.reshape(-1, 6)[:, :3]
                        if arr.ndim == 1 and len(arr) == 3*num_points: # Raw 3D
                            arr = arr.reshape(-1, 3)
                        
                        if arr.ndim > 1 and arr.shape[1] == 3: is_vector = True
                        flat_data.append(arr.astype(np.float32))
                    
                    combined = np.concatenate(flat_data, axis=0)
                    ds = pd.create_dataset(field, data=combined)
                    if is_vector:
                        ds.attrs.create("Units", np.bytes_("mm"))
            else:
                # Single step logic
                for field, data in self.results.items():
                    if not field.endswith("_el") and len(data) == num_points:
                        pd.create_dataset(field, data=data.astype(np.float32))
                    elif (field.endswith("_vec") or field == 'displacement_vec') and data.ndim == 1 and len(data) == 6*num_points:
                        vec_3d = np.asarray(data).reshape(-1, 6)[:, :3].astype(np.float32)
                        pd.create_dataset(field, data=vec_3d)
        
        self.last_saved = actual_path
        print(f" -> Result saved to: {actual_path}")
        return self

    def save_glb(self, filepath):
        """Exports the deformed result to a modern GLB file for web/presentation."""
        grid = self._get_pv_grid(scale=30.0) # Standard 30x scale for visual WOW
        grid.point_data["stress_vm"] = self.results.get("stress_vm", np.zeros(len(self.nodes)))
            
        p = pv.Plotter(off_screen=True)
        p.add_mesh(grid, scalars="stress_vm", cmap="coolwarm", smooth_shading=True)
        p.export_gltf(filepath) # GLB is GLTF binary
        p.close()
        return self

    def open_paraview(self, filepath=None):
        """
        Launches ParaView with an automated Python script for zero-click visualization.
        """
        target_file = filepath if filepath else getattr(self, 'last_saved', None)
        if not target_file or not os.path.exists(target_file):
            print(f" [ERROR] File not found for ParaView: {target_file}")
            return self
            
        pv_path = r"C:\Program Files\ParaView 6.0.1\bin\paraview.exe"
        if not os.path.exists(pv_path):
            print(f" [WARNING] ParaView executable not found at {pv_path}")
            return self

        # --- Generate Automation Script ---
        script_path = os.path.join(os.path.dirname(os.path.abspath(target_file)), "pv_auto_visualize.py")
        abs_target = os.path.abspath(target_file).replace("\\", "/")
        
        # Decide scaling and vector field
        field_to_color = 'stress_vm' if 'stress_vm' in self.results else list(self.results.keys())[0] if self.results else None
        vector_field = 'displacement_vec' if 'displacement_vec' in self.results else None
        if not vector_field:
            # Fallback to any field ending with _vec
            vfields = [f for f in self.results.keys() if f.endswith('_vec')]
            if vfields: vector_field = vfields[0]

        script_content = f"""
from paraview.simple import *
# 1. Load Data
reader = OpenDataFile('{abs_target}')
if not reader:
    print('Failed to open file')
else:
    UpdatePipeline()
    # 2. Setup View
    view = GetActiveViewOrCreate('RenderView')
    view.Background = [1, 1, 1] # Clean White
    
    # --- BEFORE: Original Mesh (Wireframe) ---
    original_display = Show(reader, view)
    original_display.Representation = 'Wireframe'
    original_display.DiffuseColor = [0.7, 0.7, 0.7] # Subtle Gray
    original_display.Opacity = 0.2

    # --- AFTER: Deformed Mesh (Warp + Surface) ---
    if '{vector_field}':
        warp = WarpByVector(Input=reader)
        warp.Vectors = ['POINTS', '{vector_field}']
        # Auto-scale: roughly 10% of bbox size
        warp.ScaleFactor = 20.0 # Default factor for visual wow
        
        UpdatePipeline()
        display = Show(warp, view)
        display.Representation = 'Surface With Edges'
        
        # 3. Coloring
        if '{field_to_color}':
            ColorBy(display, ('POINTS', '{field_to_color}'))
            lut = GetColorTransferFunction('{field_to_color}')
            lut.ApplyPreset('Cool to Warm', (True))
            display.SetScalarBarVisibility(view, True)
    else:
        # Fallback to simple show if no vectors
        display = Show(reader, view)
        display.Representation = 'Surface With Edges'

    # 4. Camera & Render
    ResetCamera()
    Render()
"""
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
            
        print(f" -> Launching ParaView with automation script: {script_path}")
        try:
            subprocess.Popen([pv_path, "--script=" + script_path])
        except Exception as e:
            print(f" [ERROR] Failed to launch ParaView: {e}")
            
        return self

class PlateFEM:
    """High-level API for generalized FEM structural analysis."""
    def __init__(self, Lx=None, Ly=None, nx=None, ny=None, nodes=None, elements=None):
        if nodes is None:
            # Default to tray mesh as per project standard
            nodes, elements = generate_tray_mesh_quads(Lx, Ly, wall_width=50.0, wall_height=50.0, nx=nx, ny=ny, mode='vertical')
        
        self.fem = ShellFEM(nodes, elements)
        self.Lx, self.Ly = Lx, Ly
        self.nx, self.ny = nx, ny
        
        self.constraints = [] # List of (dof_indices, values)
        self.loads = []       # List of (dof_indices, values)
        self.rbe2_links = []  # List of (master_pos, slave_select_func)
        self.entity_node_map = {} # (dim, tag) -> [node_indices]
        self.entity_bboxes = {}   # (dim, tag) -> [xmin, ymin, zmin, xmax, ymax, zmax]
        self.cad_bbox = None

    # --- Delegation for Backward Compatibility ---
    @property
    def nodes(self): return self.fem.nodes
    @property
    def elements(self): 
        # Returns quads and trias combined if possible, else list of lists
        if self.fem.trias.size == 0: return self.fem.quads
        if self.fem.quads.size == 0: return self.fem.trias
        # Heterogeneous: return list of lists
        return [list(e) for e in self.fem.quads] + [list(e) for e in self.fem.trias]
    @property
    def total_dof(self): return self.fem.total_dof
    
    def assemble(self, params, sparse=False): return self.fem.assemble(params, sparse)
    def solve_static_sparse(self, K, F, free, fixed, vals): return self.fem.solve_static_sparse(K, F, free, fixed, vals)
    def solve_eigen_sparse(self, K, M, num_modes): return self.fem.solve_eigen_sparse(K, M, num_modes)
    def solve_eigen(self, K, M, num_modes, num_skip): return self.fem.solve_eigen(K, M, num_modes, num_skip)
    def solve_eigen_arpack_jax_compatible(self, K, M, num_modes, num_skip): return self.fem.solve_eigen_arpack(K, M, num_modes=num_modes, num_skip=num_skip)
    def solve_static_partitioned(self, K, F, free, fixed, vals): return self.fem.solve_static_partitioned(K, F, free, fixed, vals)
    def compute_field_results(self, u, params): return self.fem.compute_field_results(u, params)
    def compute_max_surface_stress(self, u, params, field_results=None): return self.fem.compute_max_surface_stress(u, params, field_results)
    def compute_max_surface_strain(self, u, params, field_results=None): return self.fem.compute_max_surface_strain(u, params, field_results)
    def compute_strain_energy_density(self, u, params, field_results=None): return self.fem.compute_strain_energy_density(u, params, field_results)

    def solve_eigen_primme(self, K, M, num_modes, num_skip=6):
        """
        High-performance eigenvalue solver using PRIMME.
        Best for multi-core CPU environments and finding smallest eigenvalues.
        Requires: pip install primme
        """
        import numpy as np
        import scipy.sparse as sp
        try:
            import primme
        except ImportError:
            raise ImportError("PRIMME library is not installed. Please run: pip install primme")
            
        if hasattr(K, 'device_buffer') or 'jax' in str(type(K)): K = np.array(K)
        if hasattr(M, 'device_buffer') or 'jax' in str(type(M)): M = np.array(M)
        
        total_modes = int(num_modes) + int(num_skip)
        
        diag_M = M.diagonal() if sp.issparse(M) else np.diag(M)
        diag_M = np.maximum(diag_M, 1e-12)
        M_safe = sp.diags(diag_M) if sp.issparse(M) else np.diag(diag_M)

        vals, vecs = primme.eigsh(K, k=total_modes, M=M_safe, which='SM', tol=1e-4)
        
        idx = np.argsort(vals)
        vals_sorted, vecs_sorted = vals[idx], vecs[:, idx]
        
        return vals_sorted[num_skip : num_skip + num_modes], vecs_sorted[:, num_skip : num_skip + num_modes]

    def solve_eigen_lobpcg(self, K, M, num_modes, num_skip=6):
        """
        High-performance eigenvalue solver using SciPy's LOBPCG.
        Optimal for multi-core environments and leverages Jacobi preconditioning.
        """
        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse.linalg import lobpcg, LinearOperator
        import warnings
        
        # 1. Convert to pure NumPy
        if hasattr(K, 'device_buffer') or 'jax' in str(type(K)): K = np.array(K)
        if hasattr(M, 'device_buffer') or 'jax' in str(type(M)): M = np.array(M)
        
        total_modes = int(num_modes) + int(num_skip)
        N = K.shape[0]
        
        # 2. Scale matrices to prevent huge absolute residuals in LOBPCG
        diag_K = K.diagonal() if sp.issparse(K) else np.diag(K)
        scale_factor = 1.0 / (np.max(np.abs(diag_K)) + 1e-12)
        K_scaled = K * scale_factor
        M_scaled = M * scale_factor
        
        # 2. Initial guess (Random)
        np.random.seed(42)
        X = np.random.rand(N, total_modes)
        
        # 3. Simple Jacobi Preconditioner (Diagonal of K)
        diag_K_scaled = diag_K * scale_factor
        diag_K_safe = np.where(np.abs(diag_K_scaled) < 1e-12, 1.0, diag_K_scaled)
        
        def precond_matvec(v):
            return v / diag_K_safe if v.ndim == 1 else v / diag_K_safe[:, None]
                
        M_precond = LinearOperator((N, N), matvec=precond_matvec)
        
        # 4. Ensure M is positive definite (Add small epsilon to diagonal)
        diag_M = M_scaled.diagonal() if sp.issparse(M_scaled) else np.diag(M_scaled)
        diag_M = np.maximum(diag_M, 1e-12)
        M_safe = sp.diags(diag_M) if sp.issparse(M_scaled) else np.diag(diag_M)

        # 5. Execute LOBPCG
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals, vecs = lobpcg(K_scaled, X, B=M_safe, M=M_precond, largest=False, tol=1e-3, maxiter=800)
        
        # 6. Sort and Slice
        idx = np.argsort(vals)
        vals_sorted, vecs_sorted = vals[idx], vecs[:, idx]
        
        return vals_sorted[num_skip : num_skip + num_modes], vecs_sorted[:, num_skip : num_skip + num_modes]
    # ---------------------------------------------

    @staticmethod
    def from_cad(cad_path, mesh_size=20.0, element_type='quad', curvature_adaptation=True):
        """Creates a PlateFEM instance from a CAD file using Gmsh."""
        import gmsh
        gmsh.initialize()
        gmsh.model.add("CAD_Part")
        gmsh.model.occ.importShapes(cad_path)
        gmsh.model.occ.synchronize()
        
        if curvature_adaptation:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        
        if element_type == 'quad':
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 8) # DelQuad
        
        gmsh.model.mesh.generate(2)
        
        # 1. Store node mappings and coordinates
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = node_coords.reshape(-1, 3)
        # Create map from tag to 0-based index
        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}
        
        # 2. Extract Geometric Entity Mappings
        entity_node_map = {}
        entity_bboxes = {}
        entities = gmsh.model.getEntities()
        for dim, tag in entities:
            e_node_tags, _, _ = gmsh.model.mesh.getNodes(dim, tag)
            entity_node_map[(dim, tag)] = [tag_to_idx[int(t)] for t in e_node_tags if int(t) in tag_to_idx]
            entity_bboxes[(dim, tag)] = gmsh.model.getBoundingBox(dim, tag)
            
        cad_bbox = gmsh.model.getBoundingBox(-1, -1)
        
        # 3. Handle Element types: 2 for Triangle (3 nodes), 3 for Quad (4 nodes)
        trias_list = []
        quads_list = []
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2) # 2D elements
        for t_type, tags in zip(elem_types, elem_node_tags):
            if t_type == 2: # 3-node triangle
                trias_list.append(tags.reshape(-1, 3))
            elif t_type == 3: # 4-node quad
                quads_list.append(tags.reshape(-1, 4))
            
        # Convert elem node tags to 0-based indices
        def map_tags_to_indices(elem_list):
            if not elem_list: return np.zeros((0, elem_list[0].shape[1] if elem_list else 3), dtype=np.int32)
            raw = np.concatenate(elem_list)
            indices = []
            for row in raw:
                indices.append([tag_to_idx[int(t)] for t in row])
            return np.array(indices, dtype=np.int32)
        
        final_trias = map_tags_to_indices(trias_list) if trias_list else None
        final_quads = map_tags_to_indices(quads_list) if quads_list else None
        
        gmsh.finalize()
        
        inst = PlateFEM(nodes=nodes, elements=final_quads if final_quads is not None else final_trias)
        # Manually set both if both exist
        if final_trias is not None and final_quads is not None:
             inst.fem = ShellFEM(nodes, quads=final_quads, trias=final_trias)

        inst.entity_node_map = entity_node_map
        inst.entity_bboxes = entity_bboxes
        inst.cad_bbox = cad_bbox
        return inst

    @property
    def node_coords(self): return self.fem.nodes
    @property
    def total_dof(self): return self.fem.total_dof

    def get_cad_bbox(self):
        """Returns the global bounding box of the CAD model."""
        return self.cad_bbox

    def get_entity_bbox(self, dim, tag):
        """Returns the bounding box of a specific entity."""
        return self.entity_bboxes.get((dim, tag))

    def find_nearest_entity(self, pos, dim=None):
        """
        Finds the nearest entity to a 3D position based on entity bounding boxes.
        Returns (dim, tag) and distance.
        """
        min_dist = float('inf')
        nearest = None
        pos = np.array(pos)
        
        target_entities = self.entity_bboxes.keys()
        if dim is not None:
            target_entities = [e for e in target_entities if e[0] == dim]
            
        for d, t in target_entities:
            bbox = self.entity_bboxes[(d, t)]
            # Center of BBox
            center = np.array([(bbox[0]+bbox[3])/2.0, (bbox[1]+bbox[4])/2.0, (bbox[2]+bbox[5])/2.0])
            dist = np.linalg.norm(pos - center)
            if dist < min_dist:
                min_dist = dist
                nearest = (d, t)
        
        return nearest, min_dist

    def get_entity_nodes(self, dim, tag):
        """Returns zero-based node indices associated with a geometric entity."""
        return self.entity_node_map.get((dim, tag), [])

    def add_constraint_on_entity(self, dim, tag, dofs=[0,1,2,3,4,5], value=0.0):
        """Adds constraints directly to an edge, face, or vertex."""
        target_nodes = self.get_entity_nodes(dim, tag)
        for node in target_nodes:
            for d in dofs:
                self.constraints.append((int(node*6+d), float(value)))

    def add_force_on_entity(self, dim, tag, dof=2, value=-1.0, is_total=False):
        """Adds loads directly to an edge, face, or vertex."""
        target_nodes = self.get_entity_nodes(dim, tag)
        if not target_nodes: return
        
        val_per_node = float(value) / len(target_nodes) if is_total else float(value)
        for node in target_nodes:
            self.loads.append((int(node*6+dof), val_per_node))

    def clear_bcs(self):
        """Resets all constraints and loads for a clean analysis run."""
        self.constraints = []
        self.loads = []

    def export_bcs(self):
        """Returns the current BCs/Loads as (fixed_dofs, fixed_vals, F_vec)."""
        fixed_dofs = jnp.array([c[0] for c in self.constraints], dtype=jnp.int32)
        fixed_vals = jnp.array([c[1] for c in self.constraints])
        
        F = jnp.zeros(self.fem.total_dof)
        # Use JAX-friendly assembly for the force vector if possible, 
        # but here we use a simple loop as it's usually small.
        for d, v in self.loads:
            F = F.at[d].add(v)
            
        return fixed_dofs, fixed_vals, F

    def add_constraint(self, x_range=None, y_range=None, z_range=None, dofs=[0,1,2,3,4,5], value=0.0):
        target_nodes = mesh_utils.get_nodes_in_box(self.fem.nodes, x_range, y_range, z_range)
        for node in target_nodes:
            for i, d in enumerate(dofs):
                v = value[i] if isinstance(value, (list, np.ndarray, jnp.ndarray, tuple)) else value
                self.constraints.append((int(node*6+d), float(v)))

    def add_constraint_field(self, x_range=None, y_range=None, z_range=None, dofs=[0,1,2,3,4,5], func=None):
        """Adds constraints where the value is a function of (x, y, z)."""
        target_nodes = mesh_utils.get_nodes_in_box(self.fem.nodes, x_range, y_range, z_range)
        coords = self.fem.nodes
        for node in target_nodes:
            x, y, z = coords[node]
            val = func(x, y, z)
            # func can return a single float or a list of values for each dof
            for i, d in enumerate(dofs):
                v = val[i] if isinstance(val, (list, np.ndarray, jnp.ndarray)) else val
                self.constraints.append((int(node*6+d), float(v)))

    def add_constraint_radius(self, center, radius, dofs=[0,1,2,3,4,5], value=0.0):
        target_nodes = mesh_utils.get_nodes_in_radius(self.fem.nodes, center, radius)
        for node in target_nodes:
            for i, d in enumerate(dofs):
                v = value[i] if isinstance(value, (list, np.ndarray, jnp.ndarray, tuple)) else value
                self.constraints.append((int(node*6+d), float(v)))

    def add_force(self, x_range=None, y_range=None, z_range=None, dof=2, value=-1.0, is_total=False):
        target_nodes = mesh_utils.get_nodes_in_box(self.fem.nodes, x_range, y_range, z_range)
        if len(target_nodes) == 0: return
        
        val_per_node = float(value) / len(target_nodes) if is_total else float(value)
        for node in target_nodes:
            self.loads.append((int(node*6+dof), val_per_node))

    def add_force_radius(self, center, radius, dof=2, value=-1.0, is_total=False):
        target_nodes = mesh_utils.get_nodes_in_radius(self.fem.nodes, center, radius)
        if len(target_nodes) == 0: return
        
        val_per_node = float(value) / len(target_nodes) if is_total else float(value)
        for node in target_nodes:
            self.loads.append((int(node*6+dof), val_per_node))

    def add_rbe2(self, master_pos, slave_range_box=None, slave_ids=None):
        """Adds RBE2 rigid connection."""
        # 1. Create/Find master node
        master_pos = np.array(master_pos)
        # Find if node already exists at pos
        dist = np.linalg.norm(self.fem.nodes - master_pos, axis=1)
        if np.min(dist) < 1e-3:
            master_idx = int(np.argmin(dist))
        else:
            # Add virtual master node
            new_nodes = np.vstack([self.fem.nodes, master_pos])
            master_idx = len(new_nodes) - 1
            # Re-init ShellFEM with new node
            self.fem = ShellFEM(new_nodes, quads=self.fem.quads, trias=self.fem.trias)
        
        # 2. Find slave nodes
        if slave_ids is None:
            if slave_range_box is not None:
                xr, yr, zr = slave_range_box
                slave_ids = mesh_utils.get_nodes_in_box(self.fem.nodes, xr, yr, zr)
            else:
                slave_ids = []
        
        # Remove master from slaves if present
        slave_ids = [s for s in slave_ids if s != master_idx]
        self.fem.add_rbe2(master_idx, slave_ids)
        return master_idx

    def solve_static(self, params, method='direct'):
        """
        Solves the static equilibrium problem (Ku = F).
        method: 'direct' (standard sparse solver) or 'cg' (Conjugate Gradient with progress)
        """
        print("\n" + "-"*50)
        print(" [ANALYSIS] STARTING STATIC SOLVE")
        print("-"*50)
        
        # 1. Assembly
        with tqdm.tqdm(total=4, desc=" -> Assembly Phases", unit="step") as pbar:
            pbar.set_postfix_str("Building K/M...")
            Kg, _ = self.fem.assemble(params, sparse=True)
            pbar.update(1)
            
            pbar.set_postfix_str("Applying Constraints...")
            fixed_dofs = np.unique([c[0] for c in self.constraints])
            fixed_vals = np.array([c[1] for c in self.constraints])
            free_dofs = np.setdiff1d(np.arange(self.fem.total_dof), fixed_dofs)
            pbar.update(1)
            
            pbar.set_postfix_str("Building F-vector...")
            F = np.zeros(self.fem.total_dof)
            for d, v in self.loads: F[d] += v
            pbar.update(1)
            
            pbar.set_postfix_str("Finalizing Sparsity...")
            # If assemble already returned CSR, this is fast; if not, we ensure it.
            K_sparse = Kg.tocsr() if hasattr(Kg, 'tocsr') else Kg
            pbar.update(1)

        # 2. Solving
        print(f" -> Solving Linear System (Method: {method})...")
        if method == 'cg':
            # Case 1: Iterative CG solver with real-time feedback
            def progress_callback(res): jax.debug.print("   >> Residual: {x:.2e}", x=res)
            # CG solve requires symmetric positive definite matrix
            u = self.fem.solve_static_cg(K_sparse, F, free_dofs, fixed_dofs, fixed_vals, callback=progress_callback)
        else:
            # Case 2: Direct sparse solver
            u = self.fem.solve_static_sparse(K_sparse, F, free_dofs, fixed_dofs, fixed_vals)
            
        print(" -> Recovering Stress/Strain Fields...")
        res_fields = self.fem.compute_field_results(u, params)
        
        elems = []
        if self.fem.quads.size > 0: elems.append(self.fem.quads)
        if self.fem.trias.size > 0: elems.append(self.fem.trias)
        
        res = StructuralResult(res_fields, np.array(self.fem.nodes), elems)
        res.results['u_full'] = np.array(u) # Store full vector
        res.results['displacement_vec'] = np.array(u) # For ParaView Warp mapping
        print(" [OK] Static solve complete.")
        return res

    def solve_eigen(self, K_or_params, M=None, num_modes=10, num_skip=0):
        """
        Performs Natural Frequency (Modal) analysis.
        Supports both high-level (params dict) and low-level (K, M matrices) calls.
        """
        if isinstance(K_or_params, dict):
            # High-level API call
            params = K_or_params
            print("\n" + "-"*50)
            print(" [ANALYSIS] STARTING MODAL SOLVE")
            print("-"*50)
            
            with tqdm.tqdm(total=4, desc=" -> Pre-analysis Phases", unit="step") as pbar:
                pbar.set_postfix_str("Assembling K/M...")
                Kg, Mg = self.fem.assemble(params, sparse=True)
                pbar.update(1)
                
                pbar.set_postfix_str("Partitioning Dofs...")
                fixed_dofs = np.unique([c[0] for c in self.constraints])
                free_dofs = np.setdiff1d(np.arange(self.total_dof), fixed_dofs)
                pbar.update(1)
                
                pbar.set_postfix_str("Converting to Dense...")
                if hasattr(Kg, 'toarray'):
                    K_free = Kg[free_dofs, :][:, free_dofs].toarray()
                    M_free = Mg[free_dofs, :][:, free_dofs].toarray()
                else:
                    K_free = np.array(Kg)[free_dofs, :][:, free_dofs]
                    M_free = np.array(Mg)[free_dofs, :][:, free_dofs]
                pbar.update(1)
                
                pbar.set_postfix_str("Spectral Launch Status...")
                pbar.update(1)
        else:
            # Low-level API call (within optimization loop)
            K_free, M_free = K_or_params, M
            # During optimization, we don't fix DOFs here as K/M are already partitioned or handled by loss_fn
            # But we need free_dofs for mapping results back
            fixed_dofs = np.unique([c[0] for c in self.constraints])
            free_dofs = np.setdiff1d(np.arange(self.total_dof), fixed_dofs)

        # Standard solver launch
        freqs, modes_free = self.fem.solve_eigen(K_free, M_free, num_modes=num_modes, num_skip=num_skip)
        
        # If this was a low-level call (within optimization), return the raw tuple for speed and unpacking
        if not isinstance(K_or_params, dict):
            return freqs, modes_free

        # --- HIGH-LEVEL ONLY: Reconstruct full mode shapes and wrap in StructuralResult ---
        num_free = len(free_dofs)
        results = {}
        elems = []
        if self.fem.quads.size > 0: elems.append(self.fem.quads)
        if self.fem.trias.size > 0: elems.append(self.fem.trias)
        
        for i in range(min(num_modes, len(freqs))):
            f_hz = float(freqs[i])
            mode_full = np.zeros(self.total_dof)
            mode_full[free_dofs] = modes_free[:, i]
            
            # Normalize mode shape
            mode_full /= (np.max(np.abs(mode_full)) + 1e-15)
            
            field_name = f"Mode_{i+1:02d}_{f_hz:.1f}Hz"
            # Store displacement magnitude of the mode
            results[field_name] = np.linalg.norm(mode_full.reshape(-1, 6)[:, :3], axis=1)
            results[f"{field_name}_vec"] = mode_full # Full 6-DOF vector
            
        print(f" [OK] Modal analysis complete. First mode: {freqs[0]:.2f} Hz")
        return StructuralResult(results, np.array(self.nodes), elems)
