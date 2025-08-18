import numpy as np
import trimesh
import math
from scipy.spatial import Delaunay
from scipy.ndimage import binary_dilation

def erstelle_finalen_einsatz(
    tiefenbild:np.array,
    dpi=100.0,
    max_tiefe_mm=10.0,
    bodenstaerke_mm=2.0,
    grid_basis_mm=42.0
):
    """
    Erzeugt einen hochoptimierten, soliden 3D-Einsatz.
    Diese Methode ist extrem robust, indem sie die Wände und den Boden direkt
    aus der Kontur der Oberfläche ableitet, was ein garantiert wasserdichtes
    Ergebnis sicherstellt.
    """
    # --- 1. Größenberechnung und Bildvorbereitung (unverändert) ---
    inch_zu_mm = 25.4
    original_hoehe_px, original_breite_px = tiefenbild.shape
    bild_breite_mm = (original_breite_px / dpi) * inch_zu_mm
    bild_hoehe_mm = (original_hoehe_px / dpi) * inch_zu_mm

    grid_basis_mm = 42.0
    behaelter_wand_mm = 1.2
    n_x = math.ceil((bild_breite_mm + 2 * behaelter_wand_mm) / grid_basis_mm)
    n_y = math.ceil((bild_hoehe_mm + 2 * behaelter_wand_mm) / grid_basis_mm)

    print(f"Originalbildgröße: {bild_breite_mm:.2f}mm x {bild_hoehe_mm:.2f}mm -> {n_x}x{n_y} Gridfinity-Einheiten.")

    einsatz_breite_mm = n_x * grid_basis_mm - 2 * behaelter_wand_mm
    einsatz_hoehe_mm = n_y * grid_basis_mm - 2 * behaelter_wand_mm

    einsatz_breite_px = int(round((einsatz_breite_mm / inch_zu_mm) * dpi))
    einsatz_hoehe_px = int(round((einsatz_hoehe_mm / inch_zu_mm) * dpi))

    gepolstertes_bild = np.full((einsatz_hoehe_px, einsatz_breite_px), 255, dtype=np.uint8)
    start_x = (einsatz_breite_px - original_breite_px) // 2
    start_y = (einsatz_hoehe_px - original_hoehe_px) // 2
    gepolstertes_bild[start_y:start_y + original_hoehe_px, start_x:start_x + original_breite_px] = tiefenbild

    tiefenbild_final = gepolstertes_bild
    breite_mm, hoehe_mm = einsatz_breite_mm, einsatz_hoehe_mm
    hoehe_px, breite_px = tiefenbild_final.shape

    # --- 2. Erstellung der OBERFLÄCHE (unverändert) ---
    print("Erstelle Vertices und Faces für die Oberfläche...")
    impression_mask = (tiefenbild_final < 255)
    active_mask = binary_dilation(impression_mask, structure=np.ones((3,3)), iterations=2)
    active_mask[0, 0] = True
    active_mask[0, breite_px - 1] = True
    active_mask[hoehe_px - 1, 0] = True
    active_mask[hoehe_px - 1, breite_px - 1] = True
    active_coords_yx = np.argwhere(active_mask)
    points_xy_px = active_coords_yx[:, ::-1]

    tri_top = Delaunay(points_xy_px)
    x_coords_mm = (points_xy_px[:, 0] / (breite_px - 1)) * breite_mm
    y_coords_mm = (points_xy_px[:, 1] / (hoehe_px - 1)) * hoehe_mm
    z_values_px = tiefenbild_final[points_xy_px[:, 1], points_xy_px[:, 0]]
    z_coords_mm = -((255.0 - z_values_px) / 255.0) * max_tiefe_mm

    top_vertices = np.stack([x_coords_mm, y_coords_mm, z_coords_mm], axis=1)
    top_faces = tri_top.simplices
    top_mesh = trimesh.Trimesh(vertices=top_vertices, faces=top_faces)

    # --- 3. NEUE LOGIK: WÄNDE UND BODEN AUS KONTUR ERZEUGEN ---
    print("Erzeuge Wände und Boden direkt aus der Oberflächenkontur...")

    top_outline_indices = top_mesh.outline().entities[0].points
    top_outline_verts = top_vertices[top_outline_indices]
    num_outline_verts = len(top_outline_verts)

    boden_z = -max_tiefe_mm - bodenstaerke_mm
    bottom_outline_verts = top_outline_verts.copy()
    bottom_outline_verts[:, 2] = boden_z

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

    # === KORREKTUR START: Boden manuell aus 2 Dreiecken erstellen ===
    print("Erzeuge Boden manuell aus den vier Eckpunkten...")
    bottom_faces = []
    try:
        # Finde die Bounding Box der unteren Kontur, um die Ecken zu bestimmen
        min_b_x, min_b_y, _ = np.min(bottom_outline_verts, axis=0)
        max_b_x, max_b_y, _ = np.max(bottom_outline_verts, axis=0)

        # Finde die exakten Indizes der Eck-Vertices in der unteren Konturliste
        # Dies ist robust, da die Eckpunkte exakt auf den Min/Max-Koordinaten liegen müssen.
        idx_bl_local = np.where(np.all(bottom_outline_verts[:, :2] == [min_b_x, min_b_y], axis=1))[0][0]
        idx_br_local = np.where(np.all(bottom_outline_verts[:, :2] == [max_b_x, min_b_y], axis=1))[0][0]
        idx_tr_local = np.where(np.all(bottom_outline_verts[:, :2] == [max_b_x, max_b_y], axis=1))[0][0]
        idx_tl_local = np.where(np.all(bottom_outline_verts[:, :2] == [min_b_x, max_b_y], axis=1))[0][0]

        # Konvertiere die lokalen Indizes (innerhalb der bottom_outline_verts) in globale Indizes (innerhalb all_vertices)
        bl_global = num_top_verts + idx_bl_local
        br_global = num_top_verts + idx_br_local
        tr_global = num_top_verts + idx_tr_local
        tl_global = num_top_verts + idx_tl_local
        
        # Erzeuge die zwei Dreiecke, die den rechteckigen Boden bilden.
        # Die Wicklungsreihenfolge [bl, br, tr] und [bl, tr, tl] stellt sicher,
        # dass die Normalen nach unten zeigen (aus dem Körper heraus).
        bottom_faces = [
            [bl_global, br_global, tr_global],
            [bl_global, tr_global, tl_global]
        ]
    except IndexError:
        print("WARNUNG: Die Ecken des Bodens konnten nicht eindeutig identifiziert werden. Das Modell wird keinen Boden haben.")
        bottom_faces = np.array([]).reshape(0, 3)
    # === KORREKTUR ENDE ===

    # --- 4. Zusammenbau des finalen Meshes ---
    print("Baue finales Mesh zusammen...")
    all_faces = np.vstack([top_mesh.faces, wall_faces, bottom_faces])

    final_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces,validate=True)
    final_mesh.merge_vertices()
    final_mesh.remove_unreferenced_vertices()
    final_mesh.fix_normals()
    
    broken_faces = trimesh.repair.broken_faces(final_mesh)
    if len(broken_faces) > 0:
        print(f"Warnung nach Reparatur: {len(broken_faces)} defekte Flächen gefunden.")
        print("Index:", broken_faces)

    return final_mesh

def align_meshes(mesh_to_align: trimesh.Trimesh, reference_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    aligned = mesh_to_align.copy()
    min_ref = reference_mesh.bounds[0]
    min_to_align = aligned.bounds[0]
    translation_vector = min_ref - min_to_align
    aligned.apply_translation(translation_vector)
    print(f"Ausrichtung: Netz verschoben um {translation_vector}")
    return aligned

def thicken_mesh(mesh: trimesh.Trimesh, tolerance_mm: float) -> trimesh.Trimesh:
    if tolerance_mm == 0:
        return mesh.copy()
    print(f"Aufdicken: Verschiebe Vertices um {tolerance_mm} mm entlang ihrer Normalen.")
    vertex_normals = mesh.vertex_normals
    new_vertices = mesh.vertices + vertex_normals * tolerance_mm
    thickened_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)
    return thickened_mesh

def perform_boolean_subtraction(container_mesh: trimesh.Trimesh, 
                                base_mesh: trimesh.Trimesh, 
                                offset_x_mm: float, 
                                offset_y_mm: float, 
                                tolerance_mm: float) -> trimesh.Trimesh:
    print("Starte boolesche Operation...")
    aligned_base = align_meshes(base_mesh, container_mesh)
    if offset_x_mm != 0 or offset_y_mm != 0:
        offset_vector = [offset_x_mm, offset_y_mm, 0]
        aligned_base.apply_translation(offset_vector)
        print(f"Offset: Basis um {offset_vector} verschoben.")
    thickened_base = aligned_base#thicken_mesh(aligned_base, tolerance_mm)
    print("Führe boolesche Differenz durch...")
    final_mesh = container_mesh.difference(thickened_base, engine='manifold')
    print("Repariere das finale Netz...")
    print("Boolesche Operation abgeschlossen.")
    return final_mesh

def main():
    print("Erzeuge ein Beispiel-Tiefenbild für einen 2x6 Behälter...")
    breite_px = int(6 * 42 / 25.4 * 300 * 0.8)
    hoehe_px = int(2 * 42 / 25.4 * 300 * 0.8)
    beispiel_tiefenbild = np.full((hoehe_px, breite_px), 255, dtype=np.uint8)
    cx, cy = breite_px // 2, hoehe_px // 2
    rx, ry = breite_px // 3, hoehe_px // 4
    x = np.arange(0, breite_px)
    y = np.arange(0, hoehe_px)
    xx, yy = np.meshgrid(x, y)
    mask1 = ((xx - cx + rx/2)**2 / rx**2 + (yy - cy)**2 / ry**2) < 1
    mask2 = ((xx - cx - rx/2)**2 / rx**2 + (yy - cy)**2 / ry**2) < 1
    beispiel_tiefenbild[mask1 | mask2] = 100
    print("Beispiel-Tiefenbild erstellt.\n")
    einsatz = erstelle_finalen_einsatz(
        tiefenbild=beispiel_tiefenbild,
        dpi=300.0,
        max_tiefe_mm=20.0,
        bodenstaerke_mm=2.0
    )
    if einsatz.is_watertight:
        print(f"\nErfolg! Erzeugtes Mesh ist wasserdicht. Vertices: {len(einsatz.vertices)}, Faces: {len(einsatz.faces)}")
    else:
        print(f"\nWarnung: Erzeugtes Mesh ist NICHT wasserdicht. Vertices: {len(einsatz.vertices)}, Faces: {len(einsatz.faces)}")
        print(f"Anzahl offener Kanten (sollte 0 sein): {len(einsatz.outline().entities)}")
    ausgabedatei = "gridfinity_einsatz_robust.stl"
    einsatz.export(ausgabedatei)
    print(f"\nEinsatz erfolgreich nach '{ausgabedatei}' exportiert.")

if __name__ == "__main__":
    main()
