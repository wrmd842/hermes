# !/beegfs/documentation/programs/vudo/00_programs/72_CompositeWorkflows/00_for_beck/hermes/hermes_core/.venv/bin/python3

import numpy as np
from hermes_core import Workspace
import os
import yaml
from pathlib import Path

def create_synthetic_sphere(size=50):
    """Creates a 3D matrix with a solid sphere in the center surrounded by noise."""
    print("Generating synthetic 3D data...")
    x, y, z = np.ogrid[-size//2:size//2, -size//2:size//2, -size//2:size//2]
    mask = x**2 + y**2 + z**2 <= (size//3)**2
    
    # Create a base matrix: 50 background, 200 foreground (sphere)
    matrix = np.full((size, size, size), 50, dtype=np.uint16)
    matrix[mask] = 200
    
    # Add some random noise to test segmentation robustness
    noise = np.random.randint(-20, 20, (size, size, size), dtype=np.int16)
    matrix = np.clip(matrix + noise, 0, 255).astype(np.uint16)
    return matrix

def create_synthetic_ring(size=60, major_radius=15, minor_radius=6):
    """Creates a 3D matrix with a solid ring (torus) in the center surrounded by noise."""
    print("Generating synthetic 3D ring data...")
    
    # Create a 3D coordinate grid centered at (0,0,0)
    x, y, z = np.ogrid[-size//2:size//2, -size//2:size//2, -size//2:size//2]
    
    # Torus equation: (sqrt(x^2 + y^2) - R)^2 + z^2 <= r^2
    # R = major_radius (distance from center to middle of the tube)
    # r = minor_radius (radius of the tube itself)
    distance_from_z_axis = np.sqrt(x**2 + y**2)
    mask = (distance_from_z_axis - major_radius)**2 + z**2 <= minor_radius**2
    
    # Create a base matrix: 50 background, 200 foreground (ring)
    matrix = np.full((size, size, size), 50, dtype=np.uint16)
    matrix[mask] = 200
    
    # Add some random noise to test segmentation robustness
    noise = np.random.randint(-20, 20, (size, size, size), dtype=np.int16)
    matrix = np.clip(matrix + noise, 0, 255).astype(np.uint16)
    
    return matrix

def create_synthetic_fibers(size=60, num_fibers=15, fiber_radius=5):
    """Creates a 3D matrix with randomly oriented intersecting fibers surrounded by noise."""
    print(f"Generating synthetic 3D fiber data ({num_fibers} fibers)...")
    
    # Create a base matrix: 50 background
    matrix = np.full((size, size, size), 50, dtype=np.uint16)
    
    # Create a coordinate grid for the entire volume
    # Shape of coords: (size, size, size, 3)
    I, J, K = np.indices((size, size, size))
    coords = np.stack((I, J, K), axis=-1)
    
    for _ in range(num_fibers):
        # Pick a random point in the volume for the fiber to pass through
        p0 = np.random.rand(3) * size
        
        # Generate a random 3D direction vector for the fiber
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        
        # Calculate perpendicular distance from all voxels to the 3D line
        # Distance = ||(Point - LineOrigin) x LineDirection||
        diff = coords - p0
        cross_prod = np.cross(diff, direction)
        dist = np.linalg.norm(cross_prod, axis=-1)
        
        # Mask voxels that fall within the fiber radius
        mask = dist <= fiber_radius
        matrix[mask] = 200
        
    # Add some random noise to test segmentation robustness
    noise = np.random.randint(-20, 20, (size, size, size), dtype=np.int16)
    matrix = np.clip(matrix + noise, 0, 255).astype(np.uint16)
    
    return matrix

from scipy.ndimage import distance_transform_edt

def create_synthetic_bending_fibers(size=60, num_fibers=10, fiber_radius=5):
    """Creates a 3D matrix with curved, bending fibers using Bezier curves and EDT."""
    print(f"Generating synthetic 3D bending fiber data ({num_fibers} fibers)...")
    
    # Initialize a boolean array where True = background, False = fiber skeleton
    # (EDT calculates the distance to the nearest False value)
    skeleton_volume = np.ones((size, size, size), dtype=bool)
    
    for _ in range(num_fibers):
        # 1. Generate 4 random 3D control points for the Bezier curve
        # Scaling by size*1.5 and shifting slightly allows fibers to start/end outside the box
        p0 = (np.random.rand(3) * size * 1.5) - (size * 0.25)
        p1 = (np.random.rand(3) * size * 1.5) - (size * 0.25)
        p2 = (np.random.rand(3) * size * 1.5) - (size * 0.25)
        p3 = (np.random.rand(3) * size * 1.5) - (size * 0.25)
        
        # 2. Evaluate the curve at N points (ensuring continuous voxel lines)
        t = np.linspace(0, 1, num=size * 4)[:, np.newaxis]
        
        # 3. Apply the cubic Bezier equation
        curve = ((1-t)**3)*p0 + 3*((1-t)**2)*t*p1 + 3*(1-t)*(t**2)*p2 + (t**3)*p3
        
        # 4. Convert spatial curve coordinates to integer voxel indices
        coords = np.round(curve).astype(int)
        
        # 5. Filter out coordinates that fall outside our bounding box
        valid = (coords[:, 0] >= 0) & (coords[:, 0] < size) & \
                (coords[:, 1] >= 0) & (coords[:, 1] < size) & \
                (coords[:, 2] >= 0) & (coords[:, 2] < size)
        coords = coords[valid]
        
        # 6. Burn the valid skeleton points into the volume
        if len(coords) > 0:
            skeleton_volume[coords[:, 0], coords[:, 1], coords[:, 2]] = False
            
    # 7. Calculate distance from every voxel to the nearest skeleton voxel
    distance_map = distance_transform_edt(skeleton_volume)
    
    # 8. Create the greyscale matrix
    matrix = np.full((size, size, size), 50, dtype=np.uint16)
    
    # 9. Thicken the skeletons into solid fibers
    mask = distance_map <= fiber_radius
    matrix[mask] = 200
    
    # 10. Add random noise for realism
    noise = np.random.randint(-20, 20, (size, size, size), dtype=np.int16)
    matrix = np.clip(matrix + noise, 0, 255).astype(np.uint16)
    
    return matrix


def read_input():
    cwd = os.getcwd()
    input_file = cwd + "/" + "hermes.yaml"

    # set default values:
    data_path = ""
    out_path = ""
    data_type = ""
    voxel = 0
    output_type = ""
    segmentation = "Otsu"
    geo_props = False
    visual = False

    # Read the input file
    with open(input_file,"r") as file:
        config = yaml.safe_load(file)
        data_path = config["data"]["path"]
        data_type = config["data"]["file type"]
        voxel = config["data"]["voxel size"]
        scalar = config["data"]["scalar"]
        sampling = config["analysis"]["sampling method"]
        segmentation = config["analysis"]["segmentation"]
        segment_method = config["analysis"]["segmentation method"]
        geo_props = config["analysis"]["compute geometric properties"]
        out_path = config["export"]["path"]
        output_type= config["export"]["file type"]
        visual = config["analysis"]["visualization"]
        visual = config["analysis"]["visualization"]


    return data_path, data_type, out_path, output_type, sampling, segmentation, segment_method, geo_props, voxel, scalar, visual


# Main entrance to HERMES code 
def main():
    print("\n--- Running HERMES ---")
    # ==========================================================
    # 1. INITIALIZATION & DATA LOADING
    # ==========================================================
    
    #  Load data
    data_path, data_type, out_path, output_type, sampling, segmentation, segment_method, geo_props, voxel, scalar, visual = read_input()
    # print(data_path, data_type, out_path, output_type, segmentation, segment_method, geo_props, voxel, scalar, visual)

    # Initialize workspace with the data
    if data_type == ".vtu":
        ws = Workspace.from_vtu(data_path, voxel_size=voxel, scalars_name=scalar)
    elif data_type == "tif":
        ws = Workspace.from_file(data_path, voxel_size=voxel)
        
    # ==========================================================
    # 2. SAMPLING MODULE
    # ==========================================================
    print("\n--- Sampling ---")
    if sampling == "random":
        # extract_subvolume should be called
        pass
    elif sampling == "specific":
        # Extract a specific subvolume (e.g., zooming in on the sphere)
        sub_ws = ws.extract_subvolume(corner=(10, 10, 10), dimensions=(40, 40, 40))
        print(f"Extracted Subvolume: {sub_ws.name} | Shape: {sub_ws.matrix.shape}")
    else:
        raise ValueError("The sampling method must be either 'specific' or 'random'.")

    # # ==========================================================
    # # 3. SEGMENTATION
    # # ==========================================================
    if segmentation:
        print("\n--- Running Segmentation ---")
        ws.segment(method=segment_method, invert=False) # a variable for block size will need to be added later (for adaptive thresholding)
        print(f"Segmentation complete. Unique values in matrix: {np.unique(ws.matrix)}")

    # # ==========================================================
    # # 4. MESHING & SMOOTHING
    # # ==========================================================
    print("\n--- Meshing & Smoothing ---")
    # # Pad the matrix to ensure closed meshes on the boundaries
    ws.pad()
    
    # # Generate the initial mesh
    ws.generate_mesh()
    print(f"Initial Mesh Generated: {len(ws.vertices)} vertices, {len(ws.faces)} faces.")
    
    # # Check if mesh is a valid volume
    is_vol = ws.check_mesh()
    print(f"Is mesh a closed volume? {is_vol}")

    # # Apply Laplacian smoothing
    smoothed_mesh = ws.apply_smoothing({'laplacian': 5}) # The smoothing method will need to be user defined in the future
    print(f"Applied Smoothing. Workspace name updated to: {ws.name}")

    # # ==========================================================
    # # 5. PROPERTIES QUANTIFICATION
    # # ==========================================================
    
    if geo_props:
        print("\n--- Running Property Quantification ---")
        # Run the full analytics suite
        props = ws.compute_all_properties(fiber_sphere=10, pore_sphere=30, plane='XY', step_size=7) # this probably will have to have user defined values in the future
        print("Computed Properties:")
        for key, value in props.items():
            # Truncate long lists for cleaner console output
            if isinstance(value, list) and len(value) > 3:
                print(f"  * {key}: {value[:3]} ... (truncated)")
            else:
                print(f"  * {key}: {value}")
        
        print(f"Direction Map: {ws.direction_map}")

    # # ==========================================================
    # # 6. EXPORT / OUTPUT MODULES
    # # ==========================================================
    print("\n--- Data Exporting ---")
    
    # # 6a. Export STL
    # stl_path = "./outputs/stlFiles/test_mesh.stl"
    # ws.export_stl(stl_path)
    # print(f"Exported STL to: {stl_path}")
    if output_type == ".vtu":
        # 6b. Export VTU (for ParaView)
        vtu_path = out_path + "/" + "mesh.vtu"
        ws.export_vtu(vtu_path, scalars_name=scalar)

    if geo_props:
        # 6c. Save Properties    
        prop_path = out_path + "/" + "geometric_properties.txt"
        ws.save_properties(prop_path, append=False)
        print(f"Exported Properties to: {prop_path}")

    # # ==========================================================
    # # 7. VISUALIZATION
    # # ==========================================================
    if visual:
        print("\n--- Starting Visualization ---")
        # ws.visualize_matrix_cutoff(vmin=1, vmax=1)
        ws.visualize_matrix_cutoff_plt(vmin=1, vmax=1, downsample_factor=1)
    

if __name__ == "__main__":
    main()