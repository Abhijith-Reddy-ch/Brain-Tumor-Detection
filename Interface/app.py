# app.py
import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import io
import uuid

from model_utils import load_torch_model, predict_torch, tumor_present, compute_gradcam, load_onnx_model, predict_onnx

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"png","jpg","jpeg","bmp","tiff","gif"}

app = Flask(__name__, template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "replace-with-a-secure-key"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BACKEND = os.environ.get("INFER_BACKEND","torch").lower()

# load model at startup
if BACKEND == "onnx":
    try:
        onnx_sess, classes = load_onnx_model()
        MODEL = ("onnx", onnx_sess, classes)
        print("Loaded ONNX model.")
    except Exception as e:
        print("Failed to load ONNX model:", e)
        MODEL = ("torch",) + load_torch_model()
else:
    MODEL = ("torch",) + load_torch_model()  # ("torch", model, classes, device)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXT

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file part.")
        return redirect(url_for("index"))
    file = request.files["image"]
    if file.filename == "":
        flash("No selected file.")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Unsupported file type.")
        return redirect(url_for("index"))

    # Save uploaded file
    filename = secure_filename(file.filename)
    # create unique base to avoid collisions
    base = str(uuid.uuid4())[:8]
    saved_name = f"{base}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
    file.save(filepath)
    pil_img = Image.open(filepath).convert("RGB")

    try:
        if MODEL[0] == "torch":
            _, model, classes, device = MODEL
            res = predict_torch(pil_img, model, classes, device)
            # compute grad-cam overlay
            try:
                overlay_bgr, heatmap = compute_gradcam(model, device, pil_img, target_class=res['pred_idx'])
                overlay_name = f"{base}_overlay.png"
                overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_name)
                # overlay_bgr is BGR (cv2); convert to RGB before saving with cv2 or PIL
                overlay_rgb = overlay_bgr[..., ::-1]
                Image.fromarray(overlay_rgb).save(overlay_path)
            except Exception as e:
                overlay_name = None
                print("Grad-CAM failed:", e)
        else:
            _, sess, classes = MODEL
            res = predict_onnx(pil_img, sess, classes)
            overlay_name = None
    except Exception as e:
        flash(f"Inference error: {e}")
        return redirect(url_for("index"))

    present = tumor_present(res)
    verdict = "Tumor Detected" if present else "No Tumor Detected"
    confidence_percent = f"{res['confidence']*100:.2f}%"
    predicted_class = res['pred_class']
    probabilities = res['probabilities']

    return render_template("index.html",
                           filename=saved_name,
                           overlay=overlay_name,
                           verdict=verdict,
                           confidence=confidence_percent,
                           predicted_class=predicted_class,
                           probabilities=probabilities)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    # for local dev keep debug True; in production use gunicorn/wsgi
    app.run(host="0.0.0.0", port=5000, debug=True)
