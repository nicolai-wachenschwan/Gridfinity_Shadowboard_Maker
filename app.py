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

# Local modules (existing project modules)
import preprocess_image as pre
import make_mesh as mesh
import depth_estimation as de

st.set_page_config(layout="wide", page_title="Shadowboard Generator")

from streamlit_drawable_canvas import st_canvas

# ---------- Constants / Defaults ----------
MAX_DISPLAY_WIDTH_PX = 1100
MAX_DISPLAY_HEIGHT_PX = 700

PAPER_SIZES_MM = {"A4 (210x297)": (210, 297), "A5 (148x210)": (148, 210), "A6 (105x148)": (105, 148), "Letter (216x279)": (216, 279), "Custom": None}

DEFAULT_PARAMS = {
    "dpi": 100, "paper_format_name": "A4 (210x297)", "custom_paper_width_mm": 210, "custom_paper_height_mm": 297,
    "thickening_mm": 1, "circle_diameter_mm": 10, "circle_grayscale_value": 0,
    "height_mm": 17.0, "remaining_thickness_mm": 2.0, # Replaces max_depth_mm and floor_thickness_mm
    "grid_size_mm": 42.0, "base_tolerance_mm": 0.3, "offset_x_mm": 0.0, "offset_y_mm": 0.0,
    "output_filename": "shadowboard.stl",
    # new params
    "use_depth_map": False, "depth_threshold": 127
}

# ---------- Helper Functions ----------
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
    raise ValueError("Unknown image type")

def numpy_from_bytes_or_pil(img_like):
    if isinstance(img_like, np.ndarray): return img_like.copy()
    pil = pil_from_bytes_or_array(img_like)
    return np.array(pil)

# ---------- App state (with history) ----------
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
        st.title("🖼️ Custom Shadowboard Generator — to Gridfinity and beyond")

    def run(self):
        self.render_sidebar()
        current_stage = self.state_manager.get_current_state()["stage"]
        if current_stage == "upload": self.render_upload_stage()
        elif current_stage == "align": self.render_align_stage()
        elif current_stage == "3d": self.render_3d_stage()
        else: st.error(f"Unknown Stage: {current_stage}")

    def render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Controls")
            col1, col2 = st.columns(2)
            with col1: st.button("↩️ Undo", on_click=self.state_manager.undo, use_container_width=True, disabled=st.session_state.current_step == 0)
            with col2: st.button("↪️ Redo", on_click=self.state_manager.redo, use_container_width=True, disabled=st.session_state.current_step >= len(st.session_state.history) - 1)
            st.button("🔄 Global Reset", on_click=self.state_manager.reset_all, type="primary", use_container_width=True)

    def render_upload_stage(self):
        st.write("Your drawer needs a makeover? Then you might need custom inserts for your tools, but this is a lot of work.\n This app is made to speed up the process. Make a picture of your tools on a paper (on non white background), upload and get your stl.\n It is system agnostic, just upload your baseplate and it will create the matching cutout (tolerances included)\n")
        st.header("Step 1: Upload Image & Create Mask")
        st.info("Currently does not work well on smartphones.")

        with st.expander("📄 Parameters", expanded=True):
            # Replace checkbox with a radio button for depth model selection
            st.session_state.params["depth_model"] = st.radio(
                "Depth-Detection (Beta)",
                options=["None", "Small V2", "Base V2"],
                index=0 if not st.session_state.params.get("depth_model") or st.session_state.params.get("depth_model") == "None" else ["None", "Small V2", "Base V2"].index(st.session_state.params.get("depth_model")),
                help="Select a neural network to generate a depth map. 'None' uses a simple binary mask. The models require a one-time download."
            )

            # Show the depth threshold slider only if a depth model is selected
            if st.session_state.params.get("depth_model") != "None":
                st.session_state.params["depth_threshold"] = st.slider(
                    "Depth-Filter (Threshold)", min_value=0, max_value=255,
                    value=st.session_state.params.get("depth_threshold", 127),
                    help="Clamps depth values above this threshold. A smaller value results in a deeper maximum pocket (0=maximum depth)."
                )

            st.divider()
            st.session_state.params["dpi"] = st.number_input("DPI Scaling", value=st.session_state.params.get("dpi", 100), min_value=50, max_value=1200, step=10)
            st.session_state.params["thickening_mm"] = st.number_input("Contour Thickening (mm)", value=st.session_state.params.get("thickening_mm", 5), min_value=0, max_value=100, step=1)

            st.session_state.params["paper_format_name"] = st.selectbox("Select Format", options=list(PAPER_SIZES_MM.keys()), index=list(PAPER_SIZES_MM.keys()).index(st.session_state.params.get("paper_format_name", "A4 (210x297)")))
            if st.session_state.params["paper_format_name"] == "Custom":
                st.session_state.params["custom_paper_width_mm"] = st.number_input("Width (mm)", value=st.session_state.params.get("custom_paper_width_mm", 210))
                st.session_state.params["custom_paper_height_mm"] = st.number_input("Height (mm)", value=st.session_state.params.get("custom_paper_height_mm", 297))

        uploaded_file = st.file_uploader("📂 Upload File", type=['jpg', 'jpeg', 'png'], key="uploaded_file")
        input_file = uploaded_file
        state = self.state_manager.get_current_state()
        col1, col2 = st.columns(2)

        def params_changed(state):
            return state.get("last_processed_params") != st.session_state.params

        if input_file is not None:
            file_bytes = input_file.getvalue()
            need_process = (state["uploaded_image"] is None or file_bytes != state["uploaded_image"] or params_changed(state))
            if need_process:
                with st.spinner("Processing image..."):
                    np_array = np.frombuffer(file_bytes, np.uint8)
                    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        st.error("Could not decode image."); return

                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    rectified_image, _ = pre.process_and_undistort_paper(img_rgb, dpi=st.session_state.params["dpi"])

                    if rectified_image is None:
                        st.error("Paper detection failed. Please try another image."); return

                    # Updated logic to use the selected depth model
                    depth_model_selected = st.session_state.params.get("depth_model", "None")
                    final_processed_image = None

                    if depth_model_selected != "None":
                        st.write(f"Generating depth map with '{depth_model_selected}'...")
                        rectified_bgr = cv2.cvtColor(rectified_image, cv2.COLOR_RGB2BGR)
                        # Pass the selected model name to the get_depth_map function
                        depth_map = de.get_depth_map(rectified_bgr, model_name=depth_model_selected)
                        if depth_map is not None:
                            st.write("Filtering and cropping depth map...")
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
                                st.error("Could not find content in the mask for cropping.")
                    else:
                        # Fallback: Original binary mask logic
                        removed_bg = pre.remove(rectified_image)
                        bin_mask = pre.create_binary_mask(removed_bg)
                        offset_px = int(st.session_state.params["thickening_mm"] * (st.session_state.params["dpi"] / 25.4))
                        dilated_mask = pre.dilate_contours(bin_mask, offset_px)
                        cropped_mask, _ = pre.crop_to_content(dilated_mask) # Bbox is not needed here
                        if cropped_mask is not None:
                            final_processed_image = pre.add_white_border_pad(cropped_mask, 20)
                        else:
                            st.error("Could not find content in the mask for cropping.")

                    if final_processed_image is not None:
                        self.state_manager.update_state({
                            "uploaded_image": file_bytes,
                            "processed_mask": final_processed_image,
                            "last_processed_params": copy.deepcopy(st.session_state.params)
                        })

        state = self.state_manager.get_current_state()
        if state["uploaded_image"]:
            with col1:
                st.subheader("Original Image (Preview)")
                pil_orig = pil_from_bytes_or_array(state["uploaded_image"])
                pil_preview = resize_image_keep_aspect(pil_orig)
                st.image(pil_preview, use_container_width=False, width=pil_preview.size[0])
            with col2:
                st.subheader("Detected Mask / Depth Map (Preview)")
                if state["processed_mask"] is not None:
                    mask_pil = pil_from_bytes_or_array(state["processed_mask"])
                    mask_preview = resize_image_keep_aspect(mask_pil)
                    st.image(mask_preview, use_container_width=False, width=mask_preview.size[0], clamp=True)
                    if st.button("✅ Confirm Mask & Continue to Alignment"):
                        canvas_bg_np = numpy_from_bytes_or_pil(state["processed_mask"])
                        self.state_manager.update_state({"stage": "align", "canvas_background": canvas_bg_np, "rotation_angle": 0.0}); st.rerun()

    def render_align_stage(self):
        st.header("Step 2: Align Image & Draw Contours")
        if st_canvas is None:
            st.error("streamlit_drawable_canvas is missing — please install."); return

        state = self.state_manager.get_current_state()
        if state["canvas_background"] is None:
            st.warning("Please confirm a mask in Step 1 first.")
            if st.button("Back to Step 1"):
                self.state_manager.update_state({"stage": "upload"}); st.rerun()
            return

        st.subheader("Tools")
        current_angle = float(state.get("rotation_angle", 0.0))

        col_angle, col_stroke, col_color = st.columns(3)
        with col_angle:
            slider_angle = st.slider("Rotate Image (°)", -180.0, 180.0, current_angle, 0.5)
            precise_angle = st.number_input("Precise Angle (°)", value=current_angle, step=0.2, format="%.2f")
        with col_stroke:
            # Store stroke width in session state
            st.session_state.params["stroke_width_mm"] = st.number_input(
                "Stroke Width (mm)",
                min_value=0.1,
                max_value=50.0,
                value=st.session_state.params.get("stroke_width_mm", 10.0),
                step=0.1
            )
            stroke_width_mm = st.session_state.params["stroke_width_mm"]
        with col_color:
            # Store grayscale value in session state
            st.session_state.params["circle_grayscale_value"] = st.slider(
                "Grayscale Value (0=black)",
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

        # Store grid checkbox state in session state to persist it across reruns
        st.session_state.params['show_grid'] = st.checkbox(
            "Show Grid (5x5)",
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

        st.markdown(f"**Interactive Canvas (Resolution: {w} × {h}px)**")

        # Convert stroke width from mm to px
        dpi = st.session_state.params.get("dpi", 100)
        stroke_width_px = int(stroke_width_mm * (dpi / 25.4))
        st.info(f"Stroke width: {stroke_width_mm} mm corresponds to {stroke_width_px} px at {dpi} DPI.")

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

        if st.button("✅ Confirm Alignment & Continue to 3D Creation"):
            with st.spinner("Combining image and drawing..."):
                background_gray = rotated_img.copy()

                # Create an empty image for the drawing
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


                print("Final image with drawing:", final_image.shape)
                cv2.imwrite("final_image_with_drawing.jpg", final_image)
                self.state_manager.update_state({"stage": "3d", "aligned_image": rotated_img, "final_image_with_circles": final_image}); st.rerun()

    def render_3d_stage(self):
        st.header("Step 3: Generate & Download 3D Model")
        state = self.state_manager.get_current_state()
        if state["final_image_with_circles"] is None:
            st.warning("Please confirm the alignment in Step 2 first.")
            if st.button("Back to Step 2"): self.state_manager.update_state({"stage": "align"}); st.rerun()
            return

        st.info("Optional: Upload a base mesh (e.g., Gridfinity base) to be subtracted from your shadowboard.")
        base_mesh_file = st.file_uploader("Upload Base Mesh (.stl, .obj)", type=['stl', 'obj'], key="base_mesh_uploader")

        # Immediate validation of the uploaded base mesh
        if base_mesh_file is not None:
            # Check if this file has already been validated to avoid reprocessing
            if "validated_mesh_name" not in st.session_state or st.session_state.validated_mesh_name != base_mesh_file.name:
                with st.spinner(f"Checking mesh '{base_mesh_file.name}'..."):
                    try:
                        base_mesh_file.seek(0)
                        mesh_obj = trimesh.load(base_mesh_file, file_type=base_mesh_file.name.split('.')[-1])
                        st.session_state.uploaded_base_mesh_object = mesh_obj
                        st.session_state.validated_mesh_name = base_mesh_file.name
                        st.session_state.base_mesh_is_watertight = mesh_obj.is_watertight
                    except Exception as e:
                        st.error(f"Error loading or validating the base mesh: {e}")
                        # Delete old, potentially invalid data
                        if "uploaded_base_mesh_object" in st.session_state: del st.session_state.uploaded_base_mesh_object
                        if "validated_mesh_name" in st.session_state: del st.session_state.validated_mesh_name
                        if "base_mesh_is_watertight" in st.session_state: del st.session_state.base_mesh_is_watertight

            # Display status based on validation
            if "base_mesh_is_watertight" in st.session_state:
                if st.session_state.base_mesh_is_watertight:
                    st.success(f"✅ Mesh '{base_mesh_file.name}' is watertight and ready to use.")
                else:
                    st.warning(f"⚠️ Mesh '{base_mesh_file.name}' is not watertight. This can cause issues. An automatic repair will be attempted during generation. Tip: External programs like PrusaSlicer can often reliably fix such errors.")

        st.subheader("3D Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.params["height_mm"] = st.number_input(
                "Total Height (mm)",
                min_value=2.0,
                value=st.session_state.params.get("height_mm", 17.0),
                help="The complete height of the finished insert."
            )
        with col2:
            st.session_state.params["remaining_thickness_mm"] = st.number_input(
                "Floor Thickness (Remaining) (mm)",
                min_value=1.0,
                value=st.session_state.params.get("remaining_thickness_mm", 2.0),
                help="The material thickness remaining at the thinnest part of the floor."
            )

        st.session_state.params["grid_size_mm"] = st.number_input(
            "Grid Size (mm)",
            min_value=10.0,
            value=st.session_state.params.get("grid_size_mm", 42.0),
            step=1.0,
            help="The edge length of a single cell in the grid (e.g., 42mm for Gridfinity)."
        )

        with st.expander("Parameters for Boolean Operation", expanded=base_mesh_file is not None):
            col3, col4 = st.columns(2)
            with col3: st.session_state.params["offset_x_mm"] = st.number_input("Offset X (mm)", value=st.session_state.params.get("offset_x_mm", 0.0), format="%.2f")
            with col4: st.session_state.params["offset_y_mm"] = st.number_input("Offset Y (mm)", value=st.session_state.params.get("offset_y_mm", 0.0), format="%.2f")
            st.session_state.params["base_tolerance_mm"] = st.number_input("Base Tolerance (mm)", min_value=0.0, value=st.session_state.params.get("base_tolerance_mm", 0.3), format="%.2f")

        st.divider()

        if st.button("🚀 Generate / Compute 3D Model!", type="primary", use_container_width=True):
            with st.spinner("Generating 3D mesh... This may take a moment."):
                try:
                    # Step 1: Validate and calculate parameters
                    height_mm = st.session_state.params.get("height_mm", 17.0)
                    rem_thickness_mm = st.session_state.params.get("remaining_thickness_mm", 2.0)

                    if height_mm <= rem_thickness_mm:
                        st.error("Total height must be greater than floor thickness.")
                        return

                    max_depth_mm = height_mm - rem_thickness_mm
                    floor_thickness_mm = rem_thickness_mm

                    # Step 2: Always create the base container from the image
                    container_mesh = mesh.create_final_insert(
                        tiefenbild=state["final_image_with_circles"],
                        dpi=st.session_state.params["dpi"],
                        max_tiefe_mm=max_depth_mm,
                        bodenstaerke_mm=floor_thickness_mm,
                        grid_basis_mm=st.session_state.params.get("grid_size_mm", 42.0)
                    )

                    generated_mesh = container_mesh

                    # Step 2: Check if a base mesh was uploaded for the boolean operation
                    uploaded_base_mesh_file = st.session_state.get("base_mesh_uploader")
                    if uploaded_base_mesh_file is not None:
                        base_mesh_to_use = None
                        # Try to use the pre-validated mesh from the session state
                        if "uploaded_base_mesh_object" in st.session_state and st.session_state.get("validated_mesh_name") == uploaded_base_mesh_file.name:
                            base_mesh_to_use = st.session_state.uploaded_base_mesh_object

                            # If not watertight, repair
                            if not st.session_state.get("base_mesh_is_watertight", True):
                                st.info("Attempting to repair non-watertight base mesh...")
                                trimesh.repair.fill_holes(base_mesh_to_use)
                                base_mesh_to_use.remove_unreferenced_vertices()
                                if not base_mesh_to_use.is_watertight:
                                    st.error("Automatic repair of the base mesh failed. Skipping boolean operation.")
                                    base_mesh_to_use = None  # Prevents the operation
                                else:
                                    st.success("Base mesh successfully repaired.")
                        else:
                            # Fallback: Load the mesh if it's not in the state (should not happen)
                            st.warning("Could not find pre-validated mesh. Loading it again.")
                            try:
                                uploaded_base_mesh_file.seek(0)
                                base_mesh_to_use = trimesh.load(uploaded_base_mesh_file, file_type=uploaded_base_mesh_file.name.split('.')[-1])
                                if not base_mesh_to_use.is_watertight:
                                     st.warning("The mesh is not watertight. Attempting repair.")
                                     trimesh.repair.fill_holes(base_mesh_to_use)
                                     base_mesh_to_use.remove_unreferenced_vertices()
                                     if not base_mesh_to_use.is_watertight:
                                         st.error("Repair failed. Skipping boolean operation.")
                                         base_mesh_to_use = None
                            except Exception as e:
                                st.error(f"Error reloading the base mesh: {e}")
                                base_mesh_to_use = None

                        # Only perform the boolean operation if a valid mesh is available
                        if base_mesh_to_use is not None:
                            st.info("Performing boolean operation...")
                            final_mesh = mesh.perform_boolean_subtraction(
                                container_mesh=container_mesh,
                                base_mesh=base_mesh_to_use,
                                offset_x_mm=st.session_state.params["offset_x_mm"],
                                offset_y_mm=st.session_state.params["offset_y_mm"],
                                tolerance_mm=st.session_state.params["base_tolerance_mm"]
                            )
                            generated_mesh = final_mesh
                        else:
                            st.info("No valid base mesh available. Skipping boolean operation.")
                    else:
                        st.info("No base mesh uploaded. Skipping boolean operation.")

                    if not generated_mesh.is_watertight:
                        st.warning("The generated model is not completely watertight. Repair was attempted.")
                        #trimesh.repair.fill_holes(generated_mesh)
                        #generated_mesh.remove_unreferenced_vertices()

                    # Calculate grid size for display
                    depth_image = state["final_image_with_circles"]
                    dpi = st.session_state.params["dpi"]
                    grid_base_mm = st.session_state.params.get("grid_size_mm", 42.0)
                    inch_to_mm = 25.4
                    original_height_px, original_width_px = depth_image.shape
                    image_width_mm = (original_width_px / dpi) * inch_to_mm
                    image_height_mm = (original_height_px / dpi) * inch_to_mm
                    container_wall_mm = 1.2
                    n_x = math.ceil((image_width_mm + 2 * container_wall_mm) / grid_base_mm)
                    n_y = math.ceil((image_height_mm + 2 * container_wall_mm) / grid_base_mm)

                    self.state_manager.update_state({"generated_mesh": generated_mesh, "grid_cells": f"{n_x}x{n_y}"})

                except Exception as e:
                    st.error(f"Error during mesh creation: {e}")
                    st.code(traceback.format_exc())


        state = self.state_manager.get_current_state()
        if state["generated_mesh"] is not None:
            if state.get("grid_cells"):
                st.info(f"The generated part fits into a {state['grid_cells']} grid.")
            st.subheader("Interactive 3D Preview")
            pv_mesh = pv.wrap(state["generated_mesh"])
            plotter = pv.Plotter(window_size=[800, 600], border=False)
            plotter.add_mesh(pv_mesh, color='lightblue', smooth_shading=True, specular=0.5, ambient=0.3)
            plotter.view_isometric(); plotter.background_color = 'white'
            stpyvista(plotter, key="pv_viewer")
            st.subheader("Download")
            st.session_state.params["output_filename"] = st.text_input("Filename", value=st.session_state.params.get("output_filename", "shadowboard.stl"))
            with io.BytesIO() as f:
                state["generated_mesh"].export(f, file_type='stl'); f.seek(0)
                stl_data = f.read()
            st.download_button(label="📥 Download STL File", data=stl_data, file_name=st.session_state.params["output_filename"], mime="model/stl", use_container_width=True)

app_state_manager = AppState()
app_ui = AppUI(app_state_manager)
app_ui.run()
