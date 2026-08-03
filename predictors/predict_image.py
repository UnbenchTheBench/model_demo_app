import base64
import io
import os
from PIL import Image
from ultralytics import YOLO

# 1. Resolve model path at project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "vision_model.pt")

# 2. Load model via Ultralytics YOLO API at startup
try:
    vision_model = YOLO(MODEL_PATH)
    print(f"✅ Loaded Ultralytics vision model successfully from: {MODEL_PATH}")
except Exception as e:
    vision_model = None
    print(f"⚠️ Failed to load vision model at {MODEL_PATH}: {e}")


def run_image_model(image_contents, image_filename):
    """
    Decodes Dash Upload base64 image, runs YOLO segmentation inference,
    plots mask overlay, and returns formatted result dict.
    """
    if not image_contents:
        raise ValueError("No image contents were provided.")

    if vision_model is None:
        raise RuntimeError(f"Vision model file 'vision_model.pt' could not be loaded at {MODEL_PATH}.")

    # Step A: Decode base64 input image from Dash
    header, base64_str = image_contents.split(',')
    image_bytes = base64.b64decode(base64_str)

    # Step B: Convert bytes to PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Step C: Run segmentation inference
    results = vision_model(pil_image, verbose=False)
    result = results[0]  # First image result

    # Step D: Render Segmentation Overlay Image
    annotated_bgr = result.plot()             # BGR array with drawn masks
    annotated_rgb = annotated_bgr[..., ::-1]   # Convert BGR (OpenCV) to RGB
    annotated_pil = Image.fromarray(annotated_rgb)

    # Step E: Convert overlay image to Base64 string for Dash html.Img tag
    buffered = io.BytesIO()
    annotated_pil.save(buffered, format="PNG")
    annotated_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Step F: Calculate risk score / confidence
    risk_score = 0
    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        top_conf = result.boxes.conf.max().item()
        risk_score = int(top_conf * 100)
    elif hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
        risk_score = 50

    risk_score = max(0, min(100, risk_score))

    if risk_score >= 30:
        finding = f"Scan '{image_filename}' analyzed. Segmentation masks indicate localized target areas."
    else:
        finding = f"Scan '{image_filename}' analyzed. No acute segmentation mask regions detected."

    # Return key dict matching what pages/predict.py expects
    return {
        "finding": finding,
        "risk_score": risk_score,
        "model_used": "Ultralytics YOLO Segmentation Model",
        "annotated_image": annotated_base64
    }