
from stpyvista.utils import start_xvfb
start_xvfb()

import io
import copy
import traceback
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw
import streamlit as st
import trimesh
from stpyvista import stpyvista
import pyvista as pv

# Lokale Module (bestehende Projektmodule)
import preprocess_image as pre
import make_mesh as mesh
import depth_estimation as de

st.set_page_config(layout="wide", page_title="Shadowboard Generator")

from streamlit_drawable_canvas import st_canvas

# ---------- Konstanten / Defaults ----------
MAX_DISPLAY_WIDTH_PX = 1100
MAX_DISPLAY_HEIGHT_PX = 700

PAPER_SIZES_MM = {"A4 (210x297)": (210, 297), "A5 (148x210)": (148, 210), "A6 (105x148)": (105, 148), "Letter (216x279)": (216, 279), "Benutzerdefiniert": None}

DEFAULT_PARAMS = {
    "dpi": 100, "paper_format_name": "A4 (210x297)", "custom_paper_width_mm": 210, "custom_paper_height_mm": 297,
    "thickening_mm": 1, "circle_diameter_mm": 10, "circle_grayscale_value": 0,
    "height_mm": 17.0, "remaining_thickness_mm": 2.0, # Ersetzt max_depth_mm und floor_thickness_mm
    "grid_size_mm": 42.0, "base_tolerance_mm": 0.3, "offset_x_mm": 0.0, "offset_y_mm": 0.0,
    "output_filename": "shadowboard.stl",
    # new params
    "use_depth_map": False, "depth_threshold": 127
}

# ---------- Hilfsfunktionen ----------
def resize_image_keep_aspect(pil_img: Image.Image, max_w=MAX_DISPLAY_WIDTH_PX, max_h=MAX_DISPLAY_HEIGHT_PX):
    w, h = pil_img.size
    scale = min(1.0, min(max_w / w, max_h / h))
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return pil_img.resize(new_size, Image.LANCZOS)
    return pil_img.copy()

def draw_grid_on_image(np_img: np.ndarray, rows=5, cols=5, line_alpha=120):
    if len(np_img.shape) == 2:
        np_img_rgb = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    else:
        np_img_rgb = np_img
    pil = Image.fromarray(np_img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    w, h = pil.size
    for i in range(1, cols):
        x = int(i * w / cols)
        draw.line([(x,0),(x,h)], fill=(0,0,0,line_alpha), width=1)
    for j in range(1, rows):
        y = int(j * h / rows)
        draw.line([(0,y),(w,y)], fill=(0,0,0,line_alpha), width=1)
    combined = Image.alpha_composite(pil, overlay).convert("RGB")
    return np.array(combined)

def pil_from_bytes_or_array(img_like):
    if img_like is None: return None
    if isinstance(img_like, Image.Image): return img_like.copy()
    if isinstance(img_like, (bytes, bytearray)): return Image.open(io.BytesIO(img_like))
    if isinstance(img_like, np.ndarray): return Image.fromarray(img_like)
    raise ValueError("Unbekannter Bildtyp")

def numpy_from_bytes_or_pil(img_like):
    if isinstance(img_like, np.ndarray): return img_like.copy()
    pil = pil_from_bytes_or_array(img_like)
    return np.array(pil)

# ---------- App state (mit history) ----------
class AppState:
    def __init__(self):
        if 'history' not in st.session_state: self.reset_all()

    def reset_all(self):
        st.session_state.history = [self.get_initial_state()]
        st.session_state.current_step = 0
        st.session_state.params = copy.deepcopy(DEFAULT_PARAMS)
        if "uploaded_file" in st.session_state: st.session_state["uploaded_file"] = None
        if "camera_input" in st.session_state: st.session_state["camera_input"] = None
        if "base_mesh_uploader" in st.session_state: st.session_state["base_mesh_uploader"] = None


    def get_initial_state(self):
        return {"uploaded_image": None, "processed_mask": None, "aligned_image": None, "final_image_with_circles": None,
                "generated_mesh": None, "stage": "upload", "canvas_background": None, "canvas_drawing": None,
                "rotation_angle": 0.0, "last_processed_params": None}

    def get_current_state(self):
        return st.session_state.history[st.session_state.current_step]

    def update_state(self, new_state_dict):
        new_state = {**self.get_current_state(), **new_state_dict}
        st.session_state.current_step += 1
        st.session_state.history = st.session_state.history[:st.session_state.current_step]
        st.session_state.history.append(new_state)

    def replace_current_state(self, new_state_dict):
        cur = self.get_current_state()
        cur.update(new_state_dict)
        st.session_state.history[st.session_state.current_step] = cur

    def undo(self):
        if st.session_state.current_step > 0: st.session_state.current_step -= 1

    def redo(self):
        if st.session_state.current_step < len(st.session_state.history) - 1: st.session_state.current_step += 1

# ---------- UI ----------
class AppUI:
    def __init__(self, state_manager: AppState):
        self.state_manager = state_manager
        st.title("🖼️ Custom Shadowboard Generator — mit Boolean-Operation")

    def run(self):
        self.render_sidebar()
        current_stage = self.state_manager.get_current_state()["stage"]
        if current_stage == "upload": self.render_upload_stage()
        elif current_stage == "align": self.render_align_stage()
        elif current_stage == "3d": self.render_3d_stage()
        else: st.error(f"Unbekannte Stage: {current_stage}")

    def render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Steuerung")
            col1, col2 = st.columns(2)
            with col1: st.button("↩️ Undo", on_click=self.state_manager.undo, use_container_width=True, disabled=st.session_state.current_step == 0)
            with col2: st.button("↪️ Redo", on_click=self.state_manager.redo, use_container_width=True, disabled=st.session_state.current_step >= len(st.session_state.history) - 1)
            st.button("🔄 Globaler Reset", on_click=self.state_manager.reset_all, type="primary", use_container_width=True)

    def render_upload_stage(self):
        st.header("Schritt 1: Bild hochladen & Maske erstellen")
        st.info("Funktioniert aktuell nicht so gut auf Smartphones.")

        with st.expander("📄 Parameter", expanded=True):
            st.session_state.params["use_depth_map"] = st.checkbox(
                "Tiefen-Map aus neuronalem Netz verwenden (Beta)",
                value=st.session_state.params.get("use_depth_map", False),
                help="Erzeugt statt einer binären Maske eine Graustufen-Tiefen-Map für realistischere Vertiefungen. Benötigt einen einmaligen Modelldownload (~55 MB)."
            )

            if st.session_state.params["use_depth_map"]:
                st.session_state.params["depth_threshold"] = st.slider(
                    "Tiefen-Filter (Threshold)", min_value=0, max_value=255,
                    value=st.session_state.params.get("depth_threshold", 127),
                    help="Schneidet Tiefenwerte oberhalb dieses Schwellenwerts ab. Ein kleinerer Wert resultiert in einer tieferen maximalen Aushöhlung (0=maximale Tiefe)."
                )

            st.divider()
            st.session_state.params["dpi"] = st.number_input("DPI Skalierung", value=st.session_state.params.get("dpi", 100), min_value=50, max_value=1200, step=10)
            st.session_state.params["thickening_mm"] = st.number_input("Aufdickung der Kontur (mm)", value=st.session_state.params.get("thickening_mm", 5), min_value=0, max_value=100, step=1)

            st.session_state.params["paper_format_name"] = st.selectbox("Format wählen", options=list(PAPER_SIZES_MM.keys()), index=list(PAPER_SIZES_MM.keys()).index(st.session_state.params.get("paper_format_name", "A4 (210x297)")))
            if st.session_state.params["paper_format_name"] == "Benutzerdefiniert":
                st.session_state.params["custom_paper_width_mm"] = st.number_input("Breite (mm)", value=st.session_state.params.get("custom_paper_width_mm", 210))
                st.session_state.params["custom_paper_height_mm"] = st.number_input("Höhe (mm)", value=st.session_state.params.get("custom_paper_height_mm", 297))

        uploaded_file = st.file_uploader("📂 Datei hochladen", type=['jpg', 'jpeg', 'png'], key="uploaded_file")
        input_file = uploaded_file
        state = self.state_manager.get_current_state()
        col1, col2 = st.columns(2)

        def params_changed(state):
            return state.get("last_processed_params") != st.session_state.params

        if input_file is not None:
            file_bytes = input_file.getvalue()
            need_process = (state["uploaded_image"] is None or file_bytes != state["uploaded_image"] or params_changed(state))
            if need_process:
                with st.spinner("Verarbeite Bild..."):
                    np_array = np.frombuffer(file_bytes, np.uint8)
                    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        st.error("Konnte Bild nicht dekodieren."); return

                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    rectified_image, _ = pre.process_and_undistort_paper(img_rgb, dpi=st.session_state.params["dpi"])

                    if rectified_image is None:
                        st.error("Papiererkennung fehlgeschlagen. Bitte versuchen Sie ein anderes Bild."); return

                    use_depth_map = st.session_state.params.get("use_depth_map", False)
                    final_processed_image = None

                    if use_depth_map:
                        st.write("Erzeuge Tiefen-Map...")
                        rectified_bgr = cv2.cvtColor(rectified_image, cv2.COLOR_RGB2BGR)
                        depth_map = de.get_depth_map(rectified_bgr)
                        if depth_map is not None:
                            st.write("Filtere und schneide Tiefen-Map zu...")
                            # 1. Create binary mask
                            removed_bg = pre.remove(rectified_image)
                            bin_mask = pre.create_binary_mask(removed_bg)

                            # 2. Re-normalize depth values within the tool area to 0-255
                            tool_pixels = depth_map[bin_mask == 0]
                            if tool_pixels.size > 0:
                                min_val, max_val = np.min(tool_pixels), np.max(tool_pixels)
                                if max_val > min_val:
                                    normalized_pixels = ((tool_pixels - min_val) / (max_val - min_val) * 255.0)
                                    depth_map[bin_mask == 0] = normalized_pixels.astype(np.uint8)

                            # 3. Set background to white
                            depth_map[bin_mask == 255] = 255

                            # 4. Get cropping bounding box from the DILATED mask
                            offset_px = int(st.session_state.params["thickening_mm"] * (st.session_state.params["dpi"] / 25.4))
                            dilated_mask = pre.dilate_contours(bin_mask, offset_px)
                            _, bbox = pre.crop_to_content(dilated_mask)

                            if bbox:
                                x, y, w, h = bbox
                                # 5. Crop depth map AND the original binary mask
                                cropped_depth = depth_map[y:y+h, x:x+w]
                                cropped_bin_mask = bin_mask[y:y+h, x:x+w]

                                # 6. Apply threshold (clamping) ONLY on the tool pixels
                                threshold = st.session_state.params.get("depth_threshold", 127)
                                tool_area_in_crop = cropped_depth[cropped_bin_mask == 0]
                                clipped_tool_area = np.clip(tool_area_in_crop, a_min=0, a_max=threshold)
                                cropped_depth[cropped_bin_mask == 0] = clipped_tool_area

                                # 7. Add padding
                                final_processed_image = pre.add_white_border_pad(cropped_depth, 5)
                            else:
                                st.error("Konnte keinen Inhalt in der Maske für das Zuschneiden finden.")
                    else:
                        # Fallback: Originale binäre Masken-Logik
                        removed_bg = pre.remove(rectified_image)
                        bin_mask = pre.create_binary_mask(removed_bg)
                        offset_px = int(st.session_state.params["thickening_mm"] * (st.session_state.params["dpi"] / 25.4))
                        dilated_mask = pre.dilate_contours(bin_mask, offset_px)
                        cropped_mask, _ = pre.crop_to_content(dilated_mask) # Bbox wird hier nicht benötigt
                        if cropped_mask is not None:
                            final_processed_image = pre.add_white_border_pad(cropped_mask, 20)
                        else:
                            st.error("Konnte keinen Inhalt in der Maske für das Zuschneiden finden.")

                    if final_processed_image is not None:
                        self.state_manager.update_state({
                            "uploaded_image": file_bytes,
                            "processed_mask": final_processed_image,
                            "last_processed_params": copy.deepcopy(st.session_state.params)
                        })

        state = self.state_manager.get_current_state()
        if state["uploaded_image"]:
            with col1:
                st.subheader("Originalbild (Preview)")
                pil_orig = pil_from_bytes_or_array(state["uploaded_image"])
                pil_preview = resize_image_keep_aspect(pil_orig)
                st.image(pil_preview, use_container_width=False, width=pil_preview.size[0])
            with col2:
                st.subheader("Erkannte Maske / Tiefen-Map (Preview)")
                if state["processed_mask"] is not None:
                    mask_pil = pil_from_bytes_or_array(state["processed_mask"])
                    mask_preview = resize_image_keep_aspect(mask_pil)
                    st.image(mask_preview, use_container_width=False, width=mask_preview.size[0], clamp=True)
                    if st.button("✅ Maske bestätigen & weiter zum Ausrichten"):
                        canvas_bg_np = numpy_from_bytes_or_pil(state["processed_mask"])
                        self.state_manager.update_state({"stage": "align", "canvas_background": canvas_bg_np, "rotation_angle": 0.0}); st.rerun()

    def render_align_stage(self):
        st.header("Schritt 2: Bild ausrichten & Konturen zeichnen")
        if st_canvas is None:
            st.error("streamlit_drawable_canvas fehlt — bitte installieren."); return

        state = self.state_manager.get_current_state()
        if state["canvas_background"] is None:
            st.warning("Bitte zuerst eine Maske in Schritt 1 bestätigen.")
            if st.button("Zurück zu Schritt 1"):
                self.state_manager.update_state({"stage": "upload"}); st.rerun()
            return

        st.subheader("Werkzeuge")
        current_angle = float(state.get("rotation_angle", 0.0))

        col_angle, col_stroke, col_color = st.columns(3)
        with col_angle:
            slider_angle = st.slider("Bild drehen (°)", -180.0, 180.0, current_angle, 0.5)
            precise_angle = st.number_input("Genauer Winkel (°)", value=current_angle, step=0.1, format="%.2f")
        with col_stroke:
            # Zustand der Strichstärke im Session-State speichern
            st.session_state.params["stroke_width_mm"] = st.number_input(
                "Strichstärke (mm)",
                min_value=0.1,
                max_value=50.0,
                value=st.session_state.params.get("stroke_width_mm", 5.0),
                step=0.1
            )
            stroke_width_mm = st.session_state.params["stroke_width_mm"]
        with col_color:
            # Zustand des Grauwerts im Session-State speichern
            st.session_state.params["circle_grayscale_value"] = st.slider(
                "Grauwert (0=schwarz)",
                min_value=0,
                max_value=255,
                value=st.session_state.params.get("circle_grayscale_value", 0)
            )
            g = st.session_state.params["circle_grayscale_value"]
            stroke_color = f"rgb({g}, {g}, {g})"


        if precise_angle != current_angle: new_angle = precise_angle
        else: new_angle = slider_angle
        if new_angle != current_angle:
            self.state_manager.replace_current_state({"rotation_angle": float(new_angle)}); st.rerun()

        # Grid-Checkbox-Zustand im Session-State speichern, um ihn über Reruns hinweg zu erhalten
        st.session_state.params['show_grid'] = st.checkbox(
            "Gitter einblenden (5x5)",
            value=st.session_state.params.get('show_grid', False)
        )
        show_grid = st.session_state.params['show_grid']
        st.divider()

        canvas_bg_np = numpy_from_bytes_or_pil(state["canvas_background"])
        h, w = canvas_bg_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -current_angle, 1.0)
        rotated_img = cv2.warpAffine(canvas_bg_np, M, (w, h), borderValue=255)

        canvas_for_canvas = rotated_img.copy()
        if show_grid: canvas_for_canvas = draw_grid_on_image(rotated_img, rows=5, cols=5, line_alpha=120)

        st.markdown(f"**Interaktiver Canvas (Auflösung: {w} × {h}px)**")

        # Konvertiere Strichstärke von mm in px
        dpi = st.session_state.params.get("dpi", 100)
        stroke_width_px = int(stroke_width_mm * (dpi / 25.4))
        st.info(f"Strichstärke: {stroke_width_mm} mm entspricht {stroke_width_px} px bei {dpi} DPI.")

        pil_bg_for_canvas = pil_from_bytes_or_array(canvas_for_canvas)
        canvas_result = st_canvas(
            drawing_mode="freedraw",
            stroke_width=stroke_width_px,
            stroke_color=stroke_color,
            background_image=pil_bg_for_canvas,
            update_streamlit=True,
            height=h,
            width=w,
            key="align_canvas_native"
        )

        if st.button("✅ Ausrichtung bestätigen & weiter zur 3D-Erstellung"):
            with st.spinner("Kombiniere Bild und Zeichnung..."):
                background_gray = rotated_img.copy()

                # Erstelle ein leeres Bild für die Zeichnung
                drawing_mask = np.full_like(background_gray, 255) # Start with a white mask

                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data.get("objects", [])
                    for obj in objects:
                        if obj['type'] == 'path':
                            path = [np.array(p[1:3]).astype(np.int32) for p in obj['path']]
                            cv2.polylines(drawing_mask, [np.array(path)], isClosed=False, color=(0,0,0), thickness=stroke_width_px)

                # Combine the drawing (black lines) with the background image (white/gray)
                # Where the mask is black (0), the final image should be black.
                # Where the mask is white (255), the final image should have the value of the background.
                final_image = np.minimum(background_gray, drawing_mask)


                print("Finales Bild mit Zeichnung:", final_image.shape)
                cv2.imwrite("final_image_with_drawing.jpg", final_image)
                self.state_manager.update_state({"stage": "3d", "aligned_image": rotated_img, "final_image_with_circles": final_image}); st.rerun()

    def render_3d_stage(self):
        st.header("Schritt 3: 3D-Modell generieren & herunterladen")
        state = self.state_manager.get_current_state()
        if state["final_image_with_circles"] is None:
            st.warning("Bitte zuerst die Ausrichtung in Schritt 2 bestätigen.")
            if st.button("Zurück zu Schritt 2"): self.state_manager.update_state({"stage": "align"}); st.rerun()
            return

        st.info("Optional: Laden Sie ein Basis-Mesh hoch (z.B. Gridfinity-Boden), mit dem Ihr Shadowboard verschnitten werden soll.")
        base_mesh_file = st.file_uploader("Basis-Mesh hochladen (.stl, .obj)", type=['stl', 'obj'], key="base_mesh_uploader")

        # Sofortige Validierung des hochgeladenen Base-Mesh
        if base_mesh_file is not None:
            # Prüfen, ob diese Datei bereits validiert wurde, um wiederholte Verarbeitung zu vermeiden
            if "validated_mesh_name" not in st.session_state or st.session_state.validated_mesh_name != base_mesh_file.name:
                with st.spinner(f"Prüfe Mesh '{base_mesh_file.name}'..."):
                    try:
                        base_mesh_file.seek(0)
                        mesh_obj = trimesh.load(base_mesh_file, file_type=base_mesh_file.name.split('.')[-1])
                        st.session_state.uploaded_base_mesh_object = mesh_obj
                        st.session_state.validated_mesh_name = base_mesh_file.name
                        st.session_state.base_mesh_is_watertight = mesh_obj.is_watertight
                    except Exception as e:
                        st.error(f"Fehler beim Laden oder Validieren des Base-Mesh: {e}")
                        # Alte, möglicherweise ungültige Daten löschen
                        if "uploaded_base_mesh_object" in st.session_state: del st.session_state.uploaded_base_mesh_object
                        if "validated_mesh_name" in st.session_state: del st.session_state.validated_mesh_name
                        if "base_mesh_is_watertight" in st.session_state: del st.session_state.base_mesh_is_watertight

            # Status basierend auf der Validierung anzeigen
            if "base_mesh_is_watertight" in st.session_state:
                if st.session_state.base_mesh_is_watertight:
                    st.success(f"✅ Mesh '{base_mesh_file.name}' ist wasserdicht (watertight) und bereit zur Verwendung.")
                else:
                    st.warning(f"⚠️ Mesh '{base_mesh_file.name}' ist nicht wasserdicht. Dies kann zu Problemen führen. Bei der Generierung wird eine automatische Reparatur versucht. Tipp: Externe Programme wie PrusaSlicer können solche Fehler oft zuverlässig reparieren.")

        st.subheader("3D Parameter")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.params["height_mm"] = st.number_input(
                "Gesamthöhe (mm)",
                min_value=2.0,
                value=st.session_state.params.get("height_mm", 17.0),
                help="Die komplette Höhe des fertigen Einsatzes."
            )
        with col2:
            st.session_state.params["remaining_thickness_mm"] = st.number_input(
                "Bodenstärke (Reststärke) (mm)",
                min_value=1.0,
                value=st.session_state.params.get("remaining_thickness_mm", 2.0),
                help="Die Materialstärke, die an der dünnsten Stelle des Bodens übrig bleibt."
            )

        st.session_state.params["grid_size_mm"] = st.number_input(
            "Grid-Größe (mm)",
            min_value=10.0,
            value=st.session_state.params.get("grid_size_mm", 42.0),
            step=1.0,
            help="Die Kantenlänge einer einzelnen Zelle des Rasters (z.B. 42mm für Gridfinity)."
        )

        with st.expander("Parameter für Boolean-Operation", expanded=base_mesh_file is not None):
            col3, col4 = st.columns(2)
            with col3: st.session_state.params["offset_x_mm"] = st.number_input("Offset X (mm)", value=st.session_state.params.get("offset_x_mm", 0.0), format="%.2f")
            with col4: st.session_state.params["offset_y_mm"] = st.number_input("Offset Y (mm)", value=st.session_state.params.get("offset_y_mm", 0.0), format="%.2f")
            st.session_state.params["base_tolerance_mm"] = st.number_input("Toleranz zu Basis (mm)", min_value=0.0, value=st.session_state.params.get("base_tolerance_mm", 0.3), format="%.2f")

        st.divider()

        if st.button("🚀 3D-Modell generieren / verrechnen!", type="primary", use_container_width=True):
            with st.spinner("Erzeuge 3D-Mesh... Dies kann einen Moment dauern."):
                try:
                    # Schritt 1: Parameter validieren und berechnen
                    height_mm = st.session_state.params.get("height_mm", 17.0)
                    rem_thickness_mm = st.session_state.params.get("remaining_thickness_mm", 2.0)

                    if height_mm <= rem_thickness_mm:
                        st.error("Die Gesamthöhe muss größer als die Bodenstärke sein.")
                        return

                    max_tiefe_mm = height_mm - rem_thickness_mm
                    bodenstaerke_mm = rem_thickness_mm

                    # Schritt 2: Immer den Basis-Container aus dem Bild erstellen
                    container_mesh = mesh.erstelle_finalen_einsatz(
                        tiefenbild=state["final_image_with_circles"],
                        dpi=st.session_state.params["dpi"],
                        max_tiefe_mm=max_tiefe_mm,
                        bodenstaerke_mm=bodenstaerke_mm,
                        grid_basis_mm=st.session_state.params.get("grid_size_mm", 42.0)
                    )

                    generated_mesh = container_mesh

                    # Schritt 2: Prüfen, ob ein Basis-Mesh für die Boolean-Operation hochgeladen wurde
                    uploaded_base_mesh_file = st.session_state.get("base_mesh_uploader")
                    if uploaded_base_mesh_file is not None:
                        base_mesh_to_use = None
                        # Versuche, das vorab validierte Mesh aus dem Session State zu verwenden
                        if "uploaded_base_mesh_object" in st.session_state and st.session_state.get("validated_mesh_name") == uploaded_base_mesh_file.name:
                            base_mesh_to_use = st.session_state.uploaded_base_mesh_object

                            # Wenn nicht wasserdicht, reparieren
                            if not st.session_state.get("base_mesh_is_watertight", True):
                                st.info("Versuche, das nicht-wasserdichte Basis-Mesh zu reparieren...")
                                trimesh.repair.fill_holes(base_mesh_to_use)
                                base_mesh_to_use.remove_unreferenced_vertices()
                                if not base_mesh_to_use.is_watertight:
                                    st.error("Die automatische Reparatur des Basis-Mesh ist fehlgeschlagen. Die Boolean-Operation wird übersprungen.")
                                    base_mesh_to_use = None  # Verhindert die Operation
                                else:
                                    st.success("Basis-Mesh erfolgreich repariert.")
                        else:
                            # Fallback: Lade das Mesh, wenn es nicht im State ist (sollte nicht passieren)
                            st.warning("Konnte vorab validiertes Mesh nicht finden. Lade es erneut.")
                            try:
                                uploaded_base_mesh_file.seek(0)
                                base_mesh_to_use = trimesh.load(uploaded_base_mesh_file, file_type=uploaded_base_mesh_file.name.split('.')[-1])
                                if not base_mesh_to_use.is_watertight:
                                     st.warning("Das Mesh ist nicht wasserdicht. Reparatur wird versucht.")
                                     trimesh.repair.fill_holes(base_mesh_to_use)
                                     base_mesh_to_use.remove_unreferenced_vertices()
                                     if not base_mesh_to_use.is_watertight:
                                         st.error("Reparatur fehlgeschlagen. Boolean-Operation wird übersprungen.")
                                         base_mesh_to_use = None
                            except Exception as e:
                                st.error(f"Fehler beim erneuten Laden des Base-Mesh: {e}")
                                base_mesh_to_use = None

                        # Führe die Boolean-Operation nur durch, wenn ein valides Mesh vorhanden ist
                        if base_mesh_to_use is not None:
                            st.info("Führe Boolean-Operation durch...")
                            final_mesh = mesh.perform_boolean_subtraction(
                                container_mesh=container_mesh,
                                base_mesh=base_mesh_to_use,
                                offset_x_mm=st.session_state.params["offset_x_mm"],
                                offset_y_mm=st.session_state.params["offset_y_mm"],
                                tolerance_mm=st.session_state.params["base_tolerance_mm"]
                            )
                            generated_mesh = final_mesh
                        else:
                            st.info("Kein valides Basis-Mesh vorhanden. Überspringe Boolean-Operation.")
                    else:
                        st.info("Kein Basis-Mesh hochgeladen. Überspringe Boolean-Operation.")

                    if not generated_mesh.is_watertight:
                        st.warning("Das generierte Modell ist nicht vollständig wasserdicht. Reparatur wurde versucht.")
                        #trimesh.repair.fill_holes(generated_mesh)
                        #generated_mesh.remove_unreferenced_vertices()

                    # Gittergröße für die Anzeige berechnen
                    tiefenbild = state["final_image_with_circles"]
                    dpi = st.session_state.params["dpi"]
                    grid_basis_mm = st.session_state.params.get("grid_size_mm", 42.0)
                    inch_zu_mm = 25.4
                    original_hoehe_px, original_breite_px = tiefenbild.shape
                    bild_breite_mm = (original_breite_px / dpi) * inch_zu_mm
                    bild_hoehe_mm = (original_hoehe_px / dpi) * inch_zu_mm
                    behaelter_wand_mm = 1.2
                    n_x = math.ceil((bild_breite_mm + 2 * behaelter_wand_mm) / grid_basis_mm)
                    n_y = math.ceil((bild_hoehe_mm + 2 * behaelter_wand_mm) / grid_basis_mm)

                    self.state_manager.update_state({"generated_mesh": generated_mesh, "grid_cells": f"{n_x}x{n_y}"})

                except Exception as e:
                    st.error(f"Fehler bei der Mesh-Erstellung: {e}")
                    st.code(traceback.format_exc())


        state = self.state_manager.get_current_state()
        if state["generated_mesh"] is not None:
            if state.get("grid_cells"):
                st.info(f"Das generierte Teil passt in ein {state['grid_cells']} Raster.")
            st.subheader("Interaktive 3D-Vorschau")
            pv_mesh = pv.wrap(state["generated_mesh"])
            plotter = pv.Plotter(window_size=[800, 600], border=False)
            plotter.add_mesh(pv_mesh, color='lightblue', smooth_shading=True, specular=0.5, ambient=0.3)
            plotter.view_isometric(); plotter.background_color = 'white'
            stpyvista(plotter, key="pv_viewer")
            st.subheader("Download")
            st.session_state.params["output_filename"] = st.text_input("Dateiname", value=st.session_state.params.get("output_filename", "shadowboard.stl"))
            with io.BytesIO() as f:
                state["generated_mesh"].export(f, file_type='stl'); f.seek(0)
                stl_data = f.read()
            st.download_button(label="📥 STL-Datei herunterladen", data=stl_data, file_name=st.session_state.params["output_filename"], mime="model/stl", use_container_width=True)

app_state_manager = AppState()
app_ui = AppUI(app_state_manager)
app_ui.run()
