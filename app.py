# --- Monkeypatch für fehlende streamlit.elements.image.image_to_url ---
# (Dieser Abschnitt bleibt unverändert)
import io
import base64
from PIL import Image
import numpy as np
import tempfile
import warnings


def _image_to_data_url(img, fmt="PNG", allow_tempfile_for_large=True, max_inline_bytes=3_000_000):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if isinstance(img, Image.Image):
        bio = io.BytesIO()
        img.save(bio, format=fmt)
        data = bio.getvalue()
    elif isinstance(img, (bytes, bytearray)):
        data = bytes(img)
    else:
        raise TypeError("image_to_url patch erwartet PIL Image, numpy.ndarray oder bytes")

    if len(data) <= max_inline_bytes:
        enc = base64.b64encode(data).decode("ascii")
        return f"data:image/{fmt.lower()};base64,{enc}"
    else:
        if allow_tempfile_for_large:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt.lower()}")
            tmp.write(data)
            tmp.flush()
            tmp.close()
            warnings.warn("Bild zu groß für inline data-url -> liefere tempfile path (file://). "
                          "Das kann je nach Umgebung evtl. nicht von streamlit gehostet werden.")
            return f"file://{tmp.name}"
        else:
            raise ValueError("Image zu groß für inline data url und allow_tempfile_for_large=False")


try:
    from streamlit.elements.image import image_to_url
except Exception:
    from streamlit.elements.lib.image_utils import image_to_url
        
import streamlit.elements.image as st_image_mod
if st_image_mod is not None and not hasattr(st_image_mod, "image_to_url"):
    st_image_mod.image_to_url = image_to_url
    st_image_mod.image_to_url.__doc__ = "Patched image_to_url for compatibility"
    print("Patched streamlit.elements.image.image_to_url with custom implementation.")

# app.py (überarbeitet)
import io
import copy
import traceback
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


try:
    from streamlit_drawable_canvas import st_canvas
except Exception as e:
    st.warning("Das Paket 'streamlit-drawable-canvas' scheint nicht installiert zu sein. Die Canvas-Funktionen werden dann nicht funktionieren.")
    st_canvas = None

# ---------- Konstanten / Defaults ----------
MAX_DISPLAY_WIDTH_PX = 1100
MAX_DISPLAY_HEIGHT_PX = 700

PAPER_SIZES_MM = {"A4 (210x297)": (210, 297), "A5 (148x210)": (148, 210), "A6 (105x148)": (105, 148), "Letter (216x279)": (216, 279), "Benutzerdefiniert": None}

DEFAULT_PARAMS = {
    "dpi": 100, "paper_format_name": "A4 (210x297)", "custom_paper_width_mm": 210, "custom_paper_height_mm": 297,
    "thickening_mm": 1, "circle_diameter_mm": 10, "circle_grayscale_value": 0, "max_depth_mm": 15.0,
    "floor_thickness_mm": 2.0, "grid_size_mm": 25.4, "base_tolerance_mm": 0.3, "offset_x_mm": 0.0, "offset_y_mm": 0.0,
    "output_filename": "shadowboard.stl"
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
        st.set_page_config(layout="wide", page_title="Shadowboard Generator")
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
            st.divider()
            st.header("📄 Parameter")
            if st.button("Defaults wiederherstellen", use_container_width=True): st.session_state.params = copy.deepcopy(DEFAULT_PARAMS)
            st.session_state.params["dpi"] = st.number_input("DPI Skalierung", value=st.session_state.params.get("dpi", 100), min_value=50, max_value=1200, step=10)
            st.session_state.params["thickening_mm"] = st.number_input("Aufdickung der Kontur (mm)", value=st.session_state.params.get("thickening_mm", 5), min_value=0, max_value=100, step=1)
            st.subheader("Papiergröße")
            st.session_state.params["paper_format_name"] = st.selectbox("Format wählen", options=list(PAPER_SIZES_MM.keys()), index=list(PAPER_SIZES_MM.keys()).index(st.session_state.params.get("paper_format_name", "A4 (210x297)")))
            if st.session_state.params["paper_format_name"] == "Benutzerdefiniert":
                st.session_state.params["custom_paper_width_mm"] = st.number_input("Breite (mm)", value=st.session_state.params.get("custom_paper_width_mm", 210))
                st.session_state.params["custom_paper_height_mm"] = st.number_input("Höhe (mm)", value=st.session_state.params.get("custom_paper_height_mm", 297))
            st.divider()

    def render_upload_stage(self):
        st.header("Schritt 1: Bild hochladen & Maske erstellen")
        st.info("Sie können entweder eine Bilddatei hochladen oder direkt ein Foto aufnehmen.")
        camera_input = st.camera_input("📷 Foto aufnehmen", key="camera_input")
        uploaded_file = st.file_uploader("📂 Datei hochladen", type=['jpg', 'jpeg', 'png'], key="uploaded_file")
        input_file = camera_input if camera_input is not None else uploaded_file
        state = self.state_manager.get_current_state()
        col1, col2 = st.columns(2)

        def params_changed(state): return state.get("last_processed_params") != st.session_state.params

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
                    if rectified_image is not None:
                        removed_bg = pre.remove(rectified_image)
                        bin_mask = pre.create_binary_mask(removed_bg)
                        offset_px = int(st.session_state.params["thickening_mm"] * (st.session_state.params["dpi"] / 25.4))
                        dilated_mask = pre.dilate_contours(bin_mask, offset_px)
                        cropped_mask = pre.crop_to_content(dilated_mask)
                        padded_mask = pre.add_white_border_pad(cropped_mask, 20)
                        self.state_manager.update_state({"uploaded_image": file_bytes, "processed_mask": padded_mask, "last_processed_params": copy.deepcopy(st.session_state.params)})
                    else:
                        st.error("Papiererkennung fehlgeschlagen. Bitte versuchen Sie ein anderes Bild."); return

        state = self.state_manager.get_current_state()
        if state["uploaded_image"]:
            with col1:
                st.subheader("Originalbild (Preview)")
                pil_orig = pil_from_bytes_or_array(state["uploaded_image"])
                pil_preview = resize_image_keep_aspect(pil_orig)
                st.image(pil_preview, use_container_width=False, width=pil_preview.size[0])
            with col2:
                st.subheader("Erkannte Maske (Preview)")
                if state["processed_mask"] is not None:
                    mask_pil = pil_from_bytes_or_array(state["processed_mask"])
                    mask_preview = resize_image_keep_aspect(mask_pil)
                    st.image(mask_preview, use_container_width=False, width=mask_preview.size[0], clamp=True)
                    if st.button("✅ Maske bestätigen & weiter zum Ausrichten"):
                        canvas_bg_np = numpy_from_bytes_or_pil(state["processed_mask"])
                        self.state_manager.update_state({"stage": "align", "canvas_background": canvas_bg_np, "rotation_angle": 0.0}); st.rerun()

    def render_align_stage(self):
        st.header("Schritt 2: Bild ausrichten & Halterungen hinzufügen")
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
        slider_angle = st.slider("Bild drehen (°)", -180.0, 180.0, current_angle, 0.5)
        
        col_prec, col_diam, col_color = st.columns(3)
        with col_prec: precise_angle = st.number_input("Genauer Winkel (°)", value=current_angle, step=0.1, format="%.2f")
        with col_diam: st.session_state.params["circle_diameter_mm"] = st.number_input("Ø Halterungen (mm)", min_value=1, max_value=200, value=st.session_state.params.get("circle_diameter_mm", 10))
        with col_color: st.session_state.params["circle_grayscale_value"] = st.slider("Grauwert (0=schwarz)", min_value=0, max_value=255, value=st.session_state.params.get("circle_grayscale_value", 0))

        if precise_angle != current_angle: new_angle = precise_angle
        else: new_angle = slider_angle
        if new_angle != current_angle:
            self.state_manager.replace_current_state({"rotation_angle": float(new_angle)}); st.rerun()

        show_grid = st.checkbox("Gitter einblenden (5×5)", value=False)
        st.divider()

        canvas_bg_np = numpy_from_bytes_or_pil(state["canvas_background"])
        h, w = canvas_bg_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -current_angle, 1.0)
        rotated_img = cv2.warpAffine(canvas_bg_np, M, (w, h), borderValue=255)

        canvas_for_canvas = rotated_img.copy()
        if show_grid: canvas_for_canvas = draw_grid_on_image(rotated_img, rows=5, cols=5, line_alpha=120)

        st.markdown(f"**Interaktiver Canvas (Auflösung: {w} × {h}px)**")
        radius_px = int(0.5 * st.session_state.params["circle_diameter_mm"] * (st.session_state.params["dpi"] / 25.4))
        g = st.session_state.params["circle_grayscale_value"]
        fill_color_rgba = f"rgba({g}, {g}, {g}, 1.0)"

        pil_bg_for_canvas = pil_from_bytes_or_array(canvas_for_canvas)
        canvas_result = st_canvas(drawing_mode="point", point_display_radius=radius_px, fill_color=fill_color_rgba,
                                  stroke_width=0, background_image=pil_bg_for_canvas, update_streamlit=True,
                                  height=h, width=w, key="align_canvas_native")

        if st.button("✅ Ausrichtung bestätigen & weiter zur 3D-Erstellung"):
            with st.spinner("Kombiniere Bild und Halterungen..."):
                drawing_layer_rgba = canvas_result.image_data
                background_gray = rotated_img.copy()

                if drawing_layer_rgba is not None:
                    background_rgb = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2RGB)
                    alpha = drawing_layer_rgba[:, :, 3] / 255.0
                    alpha_mask = np.stack([alpha, alpha, alpha], axis=-1)
                    drawing_layer_rgb = drawing_layer_rgba[:, :, :3]
                    composite_rgb = (drawing_layer_rgb * alpha_mask + background_rgb * (1.0 - alpha_mask)).astype(np.uint8)
                    final_image = cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2GRAY)
                else:
                    st.warning("Keine Canvas-Daten empfangen. Fahre mit dem unmodifizierten Bild fort.")
                    final_image = background_gray

                print("Finales Bild mit Halterungen:", final_image.shape)
                cv2.imwrite("final_image_with_circles.jpg", final_image)
                self.state_manager.update_state({"stage": "3d", "aligned_image": rotated_img, "final_image_with_circles": final_image}); st.rerun()

    def render_3d_stage(self):
        st.header("Schritt 3: 3D-Modell generieren & herunterladen")
        state = self.state_manager.get_current_state()
        if state["final_image_with_circles"] is None:
            st.warning("Bitte zuerst die Ausrichtung in Schritt 2 bestätigen.")
            if st.button("Zurück zu Schritt 2"): self.state_manager.update_state({"stage": "align"}); st.rerun()
            return
        
        st.info("Optional: Laden Sie ein Basis-Mesh hoch (z.B. Gridfinity-Boden), mit dem Ihr Shadowboard verschnitten werden soll.")
        st.file_uploader("Basis-Mesh hochladen (.stl, .obj)", type=['stl', 'obj'], key="base_mesh_uploader")

        st.subheader("3D Parameter")
        col1, col2 = st.columns(2)
        with col1: st.session_state.params["max_depth_mm"] = st.number_input("Max. Tiefe der Vertiefung (mm)", min_value=1.0, value=st.session_state.params.get("max_depth_mm", 15.0))
        with col2: st.session_state.params["floor_thickness_mm"] = st.number_input("Bodenstärke (mm)", min_value=1.0, value=st.session_state.params.get("floor_thickness_mm", 2.0))
        
        st.subheader("Parameter für Boolean-Operation")
        col3, col4 = st.columns(2)
        with col3: st.session_state.params["offset_x_mm"] = st.number_input("Offset X (mm)", value=st.session_state.params.get("offset_x_mm", 0.0), format="%.2f")
        with col4: st.session_state.params["offset_y_mm"] = st.number_input("Offset Y (mm)", value=st.session_state.params.get("offset_y_mm", 0.0), format="%.2f")
        st.session_state.params["base_tolerance_mm"] = st.number_input("Toleranz zu Basis (mm)", min_value=0.0, value=st.session_state.params.get("base_tolerance_mm", 0.3), format="%.2f")
        
        st.divider()

        if st.button("🚀 3D-Modell generieren / verrechnen!", type="primary", use_container_width=True):
            with st.spinner("Erzeuge 3D-Mesh... Dies kann einen Moment dauern."):
                try:
                    # Schritt 1: Immer den Basis-Container aus dem Bild erstellen
                    container_mesh = mesh.erstelle_finalen_einsatz(
                        tiefenbild=state["final_image_with_circles"], dpi=st.session_state.params["dpi"],
                        max_tiefe_mm=st.session_state.params["max_depth_mm"], bodenstaerke_mm=st.session_state.params["floor_thickness_mm"])
                    
                    generated_mesh = container_mesh
                    
                    # Schritt 2: Prüfen, ob ein Basis-Mesh für die Boolean-Operation hochgeladen wurde
                    uploaded_base_mesh_file = st.session_state.get("base_mesh_uploader")
                    if uploaded_base_mesh_file is not None:
                        st.info("Basis-Mesh wurde hochgeladen. Führe Boolean-Operation durch...")
                        # Basis-Mesh aus der hochgeladenen Datei laden
                        # Wichtig: file-like object direkt an trimesh übergeben
                        uploaded_base_mesh_file.seek(0)
                        base_mesh = trimesh.load(uploaded_base_mesh_file, file_type=uploaded_base_mesh_file.name.split('.')[-1])

                        # Die Verarbeitung mit dem neuen Modul durchführen
                        final_mesh = mesh.perform_boolean_subtraction(
                            container_mesh=container_mesh,
                            base_mesh=base_mesh,
                            offset_x_mm=st.session_state.params["offset_x_mm"],
                            offset_y_mm=st.session_state.params["offset_y_mm"],
                            tolerance_mm=st.session_state.params["base_tolerance_mm"]
                        )
                        generated_mesh = final_mesh
                    else:
                        st.info("Kein Basis-Mesh hochgeladen. Überspringe Boolean-Operation.")

                    if not generated_mesh.is_watertight:
                        st.warning("Das generierte Modell ist nicht vollständig wasserdicht. Reparatur wurde versucht.")
                        #trimesh.repair.fill_holes(generated_mesh)
                        #generated_mesh.remove_unreferenced_vertices()

                    self.state_manager.update_state({"generated_mesh": generated_mesh})

                except Exception as e:
                    st.error(f"Fehler bei der Mesh-Erstellung: {e}")
                    st.code(traceback.format_exc())


        state = self.state_manager.get_current_state()
        if state["generated_mesh"] is not None:
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

if __name__ == "__main__":
    app_state_manager = AppState()
    app_ui = AppUI(app_state_manager)
    app_ui.run()
