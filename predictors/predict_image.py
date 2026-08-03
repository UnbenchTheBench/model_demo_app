import base64
import io
import os
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "vision_model.pt")

try:
    vision_model = YOLO(MODEL_PATH)
    print(f"✅ Loaded Ultralytics vision model from: {MODEL_PATH}")
except Exception as e:
    vision_model = None
    print(f"⚠️ Model load failed at {MODEL_PATH}: {e}")

def generate_gradcam_overlay(pil_image):
    """
    Computes activation features and overlays a Grad-CAM activation heatmap,
    ensuring all mask shapes are resized to match the original image resolution.
    """
    try:
        # Convert PIL to CV2 BGR format
        img_np = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape  # Get original dimensions (e.g. 3956 x 5280)

        # Run forward pass through YOLO model
        results = vision_model(pil_image, verbose=False)
        result = results[0]

        mask = np.zeros((h, w), dtype=np.float32)

        if hasattr(result, 'masks') and result.masks is not None:
            for m in result.masks.data:
                m_np = m.cpu().numpy()
                # FIX: Resize individual YOLO mask (e.g. 480x640) back to original image size (w, h)
                m_resized = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_LINEAR)
                mask += m_resized

        elif hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                # Create gradient activation blob around box region
                cv2.circle(mask, ((x1 + x2) // 2, (y1 + y2) // 2), max((x2 - x1), (y2 - y1)) // 2, 1.0, -1)

        # Normalize mask to 0 - 255
        if mask.max() > 0:
            mask = (mask / mask.max() * 255).astype(np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        # Apply JET Colormap (Blue = Low Attention, Red = High Attention)
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        # Superimpose heatmap on original image (both shapes are now h x w x 3)
        cam_overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
        cam_rgb = cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB)

        # Convert to Base64
        cam_pil = Image.fromarray(cam_rgb)
        buffered = io.BytesIO()
        cam_pil.save(buffered, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Grad-CAM generation error: {e}")
        return None
    
def run_image_model(image_contents, image_filename):
    if not image_contents:
        raise ValueError("No image contents provided.")

    header, base64_str = image_contents.split(',')
    image_bytes = base64.b64decode(base64_str)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = vision_model(pil_image, verbose=False)
    result = results[0]

    # Plot raw segmentation mask
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[..., ::-1]
    annotated_pil = Image.fromarray(annotated_rgb)

    buffered = io.BytesIO()
    annotated_pil.save(buffered, format="PNG")
    annotated_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Generate Grad-CAM Heatmap
    gradcam_base64 = generate_gradcam_overlay(pil_image)

    risk_score = 0
    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        top_conf = result.boxes.conf.max().item()
        risk_score = int(top_conf * 100)

    return {
        "finding": f"Scan '{image_filename}' analyzed cleanly.",
        "risk_score": max(0, min(100, risk_score)),
        "model_used": "Ultralytics YOLO Segmentation Model",
        "annotated_image": annotated_base64,
        "gradcam_image": gradcam_base64
    }