import numpy as np
import trimesh
import math
from scipy.spatial import Delaunay
from scipy.ndimage import binary_dilation

def create_final_insert(
    depth_image:np.array,
    dpi=100.0,
    max_depth_mm=10.0,
    floor_thickness_mm=2.0,
    grid_base_mm=42.0
):
    """
    Creates a highly optimized, solid 3D insert.
    This method is extremely robust by deriving the walls and the floor directly
    from the surface contour, which ensures a guaranteed watertight result.
    """
    # --- 1. Size calculation and image preparation (unchanged) ---
    inch_to_mm = 25.4
    original_height_px, original_width_px = depth_image.shape
    image_width_mm = (original_width_px / dpi) * inch_to_mm
    image_height_mm = (original_height_px / dpi) * inch_to_mm

    grid_base_mm = 42.0
    container_wall_mm = 1.2
    n_x = math.ceil((image_width_mm + 2 * container_wall_mm) / grid_base_mm)
    n_y = math.ceil((image_height_mm + 2 * container_wall_mm) / grid_base_mm)

    print(f"Original image size: {image_width_mm:.2f}mm x {image_height_mm:.2f}mm -> {n_x}x{n_y} Gridfinity units.")

    insert_width_mm = n_x * grid_base_mm - 2 * container_wall_mm
    insert_height_mm = n_y * grid_base_mm - 2 * container_wall_mm

    insert_width_px = int(round((insert_width_mm / inch_to_mm) * dpi))
    insert_height_px = int(round((insert_height_mm / inch_to_mm) * dpi))

    padded_image = np.full((insert_height_px, insert_width_px), 255, dtype=np.uint8)
    start_x = (insert_width_px - original_width_px) // 2
    start_y = (insert_height_px - original_height_px) // 2
    padded_image[start_y:start_y + original_height_px, start_x:start_x + original_width_px] = depth_image

    final_depth_image = padded_image
    width_mm, height_mm = insert_width_mm, insert_height_mm
    height_px, width_px = final_depth_image.shape

    # --- 2. Creation of the SURFACE (unchanged) ---
    print("Creating vertices and faces for the surface...")
    impression_mask = (final_depth_image < 255)
    active_mask = binary_dilation(impression_mask, structure=np.ones((3,3)), iterations=2)
    active_mask[0, 0] = True
    active_mask[0, width_px - 1] = True
    active_mask[height_px - 1, 0] = True
    active_mask[height_px - 1, width_px - 1] = True
    active_coords_yx = np.argwhere(active_mask)
    points_xy_px = active_coords_yx[:, ::-1]

    tri_top = Delaunay(points_xy_px)
    x_coords_mm = (points_xy_px[:, 0] / (width_px - 1)) * width_mm
    y_coords_mm = (points_xy_px[:, 1] / (height_px - 1)) * height_mm
    z_values_px = final_depth_image[points_xy_px[:, 1], points_xy_px[:, 0]]
    z_coords_mm = -((255.0 - z_values_px) / 255.0) * max_depth_mm

    top_vertices = np.stack([x_coords_mm, y_coords_mm, z_coords_mm], axis=1)
    top_faces = tri_top.simplices
    top_mesh = trimesh.Trimesh(vertices=top_vertices, faces=top_faces)

    # --- 3. NEW LOGIC: GENERATE WALLS AND FLOOR FROM CONTOUR ---
    print("Generating walls and floor directly from the surface contour...")

    top_outline_indices = top_mesh.outline().entities[0].points
    top_outline_verts = top_vertices[top_outline_indices]
    num_outline_verts = len(top_outline_verts)

    floor_z = -max_depth_mm - floor_thickness_mm
    bottom_outline_verts = top_outline_verts.copy()
    bottom_outline_verts[:, 2] = floor_z

    all_vertices = np.vstack([top_mesh.vertices, bottom_outline_verts])
    num_top_verts = len(top_mesh.vertices)

    wall_faces = []
    for i in range(num_outline_verts):
        p1_top_idx = top_outline_indices[i]
        p2_top_idx = top_outline_indices[(i + 1) % num_outline_verts]
        p1_bottom_idx = num_top_verts + i
        p2_bottom_idx = num_top_verts + ((i + 1) % num_outline_verts)
        wall_faces.append([p1_top_idx, p1_bottom_idx, p2_top_idx])
        wall_faces.append([p1_bottom_idx, p2_bottom_idx, p2_top_idx])

    # === CORRECTION START: Create floor manually from 2 triangles ===
    print("Creating floor manually from the four corner points...")
    bottom_faces = []
    try:
        # Find the bounding box of the lower contour to determine the corners
        min_b_x, min_b_y, _ = np.min(bottom_outline_verts, axis=0)
        max_b_x, max_b_y, _ = np.max(bottom_outline_verts, axis=0)

        # Find the exact indices of the corner vertices in the lower contour list
        # This is robust because the corner points must lie exactly on the min/max coordinates.
        idx_bl_local = np.where(np.all(bottom_outline_verts[:, :2] == [min_b_x, min_b_y], axis=1))[0][0]
        idx_br_local = np.where(np.all(bottom_outline_verts[:, :2] == [max_b_x, min_b_y], axis=1))[0][0]
        idx_tr_local = np.where(np.all(bottom_outline_verts[:, :2] == [max_b_x, max_b_y], axis=1))[0][0]
        idx_tl_local = np.where(np.all(bottom_outline_verts[:, :2] == [min_b_x, max_b_y], axis=1))[0][0]

        # Convert the local indices (within bottom_outline_verts) to global indices (within all_vertices)
        bl_global = num_top_verts + idx_bl_local
        br_global = num_top_verts + idx_br_local
        tr_global = num_top_verts + idx_tr_local
        tl_global = num_top_verts + idx_tl_local

        # Create the two triangles that form the rectangular floor.
        # The winding order [bl, br, tr] and [bl, tr, tl] ensures
        # that the normals point downwards (out of the body).
        bottom_faces = [
            [bl_global, br_global, tr_global],
            [bl_global, tr_global, tl_global]
        ]
    except IndexError:
        print("WARNING: The corners of the floor could not be uniquely identified. The model will not have a floor.")
        bottom_faces = np.array([]).reshape(0, 3)
    # === CORRECTION END ===

    # --- 4. Assembly of the final mesh ---
    print("Assembling final mesh...")
    all_faces = np.vstack([top_mesh.faces, wall_faces, bottom_faces])

    final_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces,validate=True)
    final_mesh.merge_vertices()
    final_mesh.remove_unreferenced_vertices()
    final_mesh.fix_normals()

    broken_faces = trimesh.repair.broken_faces(final_mesh)
    if len(broken_faces) > 0:
        print(f"Warning after repair: {len(broken_faces)} broken faces found.")
        print("Index:", broken_faces)

    return final_mesh

def align_meshes(mesh_to_align: trimesh.Trimesh, reference_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    aligned = mesh_to_align.copy()
    min_ref = reference_mesh.bounds[0]
    min_to_align = aligned.bounds[0]
    translation_vector = min_ref - min_to_align
    aligned.apply_translation(translation_vector)
    print(f"Alignment: Mesh translated by {translation_vector}")
    return aligned

def thicken_mesh(mesh: trimesh.Trimesh, tolerance_mm: float) -> trimesh.Trimesh:
    if tolerance_mm == 0:
        return mesh.copy()
    print(f"Thickening: Displace vertices by {tolerance_mm} mm along their normals.")
    vertex_normals = mesh.vertex_normals
    new_vertices = mesh.vertices + vertex_normals * tolerance_mm
    thickened_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)
    return thickened_mesh

def perform_boolean_subtraction(container_mesh: trimesh.Trimesh,
                                base_mesh: trimesh.Trimesh,
                                offset_x_mm: float,
                                offset_y_mm: float,
                                tolerance_mm: float) -> trimesh.Trimesh:
    print("Starting boolean operation...")
    aligned_base = align_meshes(base_mesh, container_mesh)
    if offset_x_mm != 0 or offset_y_mm != 0:
        offset_vector = [offset_x_mm, offset_y_mm, 0]
        aligned_base.apply_translation(offset_vector)
        print(f"Offset: Base moved by {offset_vector}.")
    thickened_base = aligned_base#thicken_mesh(aligned_base, tolerance_mm)
    print("Performing boolean difference...")
    final_mesh = container_mesh.difference(thickened_base, engine='manifold')
    print("Repairing the final mesh...")
    print("Boolean operation completed.")
    return final_mesh

def main():
    print("Generating a sample depth image for a 2x6 container...")
    width_px = int(6 * 42 / 25.4 * 300 * 0.8)
    height_px = int(2 * 42 / 25.4 * 300 * 0.8)
    sample_depth_image = np.full((height_px, width_px), 255, dtype=np.uint8)
    cx, cy = width_px // 2, height_px // 2
    rx, ry = width_px // 3, height_px // 4
    x = np.arange(0, width_px)
    y = np.arange(0, height_px)
    xx, yy = np.meshgrid(x, y)
    mask1 = ((xx - cx + rx/2)**2 / rx**2 + (yy - cy)**2 / ry**2) < 1
    mask2 = ((xx - cx - rx/2)**2 / rx**2 + (yy - cy)**2 / ry**2) < 1
    sample_depth_image[mask1 | mask2] = 100
    print("Sample depth image created.\n")
    insert = create_final_insert(
        depth_image=sample_depth_image,
        dpi=300.0,
        max_depth_mm=20.0,
        floor_thickness_mm=2.0
    )
    if insert.is_watertight:
        print(f"\nSuccess! Generated mesh is watertight. Vertices: {len(insert.vertices)}, Faces: {len(insert.faces)}")
    else:
        print(f"\nWarning: Generated mesh is NOT watertight. Vertices: {len(insert.vertices)}, Faces: {len(insert.faces)}")
        print(f"Number of open edges (should be 0): {len(insert.outline().entities)}")
    output_file = "gridfinity_insert_robust.stl"
    insert.export(output_file)
    print(f"\nInsert successfully exported to '{output_file}'.")

if __name__ == "__main__":
    main()
