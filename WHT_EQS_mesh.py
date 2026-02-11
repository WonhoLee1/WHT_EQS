# -*- coding: utf-8 -*-
import numpy as np
import jax.numpy as jnp

"""
WHT_EQS_mesh.py
Mesh loading and positional selection logic for 2D/3D Shell elements.
Supports Bounding Box and Position+Radius selection.
"""

def get_nodes_in_box(node_coords, x_range=None, y_range=None, z_range=None):
    """
    Select node indices within a specified bounding box.
    
    Parameters:
    -----------
    node_coords : ndarray (N, 3)
    x_range : tuple (min, max) or None
    y_range : tuple (min, max) or None
    z_range : tuple (min, max) or None
    """
    mask = np.ones(len(node_coords), dtype=bool)
    
    if x_range is not None:
        mask &= (node_coords[:, 0] >= x_range[0]) & (node_coords[:, 0] <= x_range[1])
    if y_range is not None:
        mask &= (node_coords[:, 1] >= y_range[0]) & (node_coords[:, 1] <= y_range[1])
    if z_range is not None:
        mask &= (node_coords[:, 2] >= z_range[0]) & (node_coords[:, 2] <= z_range[1])
        
    return np.where(mask)[0]

def get_nodes_in_radius(node_coords, center, radius):
    """
    Select node indices within a specified radius from a center point.
    
    Parameters:
    -----------
    node_coords : ndarray (N, 3)
    center : array-like (3,)
    radius : float
    """
    center = np.array(center)
    dist = np.linalg.norm(node_coords - center, axis=1)
    return np.where(dist <= radius)[0]

def load_mesh_msh(filepath):
    """
    Basic loader for Gmsh (.msh v2.2) files.
    Extracts nodes and shell elements (Tri3, Quad4).
    """
    nodes = []
    elements = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == "$Nodes":
                num_nodes = int(lines[i+1].strip())
                for j in range(num_nodes):
                    parts = lines[i+2+j].split()
                    nodes.append([float(parts[1]), float(parts[2]), float(parts[3])])
                i += num_nodes + 2
            elif line == "$Elements":
                num_elems = int(lines[i+1].strip())
                for j in range(num_elems):
                    parts = lines[i+2+j].split()
                    # Parts: [elm-number, elm-type, number-of-tags, <tags>, node-indices...]
                    elm_type = int(parts[1])
                    num_tags = int(parts[2])
                    node_indices = [int(n) - 1 for n in parts[3 + num_tags:]] # 0-indexed
                    
                    if elm_type == 2: # 3-node triangle
                        elements.append(node_indices)
                    elif elm_type == 3: # 4-node quad
                        elements.append(node_indices)
                i += num_elems + 2
            else:
                i += 1
                
        return np.array(nodes), np.array(elements)
        
    except Exception as e:
        print(f" ⚠ Error loading .msh file: {e}")
        return None, None

def load_mesh_f06(filepath):
    """
    Basic loader for Nastran (.f06) output files (GRID and elements).
    """
    # Placeholder for Nastran parsing logic
    print(f"Reading Nastran mesh: {filepath}")
    return None, None

def get_dofs_from_nodes(node_indices, dofs_per_node=6):
    """
    Expand node indices to global DOF indices.
    """
    dofs = []
    for idx in node_indices:
        for d in range(dofs_per_node):
            dofs.append(idx * dofs_per_node + d)
    return np.array(dofs)

def get_specific_dofs(node_indices, dof_offsets=[0, 1, 2], dofs_per_node=6):
    """
    Expand node indices to specific DOF indices (e.g., only translation u,v,w).
    """
    dofs = []
    for idx in node_indices:
        for offset in dof_offsets:
            dofs.append(idx * dofs_per_node + offset)
    return np.array(dofs)
