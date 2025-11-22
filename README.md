# 🧠 BrainTumorCNN — Brain MRI Tumor Classification with Grad-CAM + Web App

A complete deep-learning system for **detecting brain tumors from MRI images** using a fine-tuned **EfficientNet-B0 CNN**, with an integrated **Flask web application**,  and **ONNX export** for fast deployment.

🔍 **Classifies MRI scans into 4 categories:**
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

⚡ Achieves **~95% accuracy** on validation   
🌐 Includes a **web interface** for easy image uploads  

> ⚠️ **Disclaimer:** This project is strictly for **educational and research** purposes.  
> It must **NOT** be used for medical diagnosis or clinical decisions.

---

# 🚀 Features

### ✔ 1. Brain Tumor MRI Classification  
Fine-tuned EfficientNet-B0 on the popular Kaggle MRI dataset.

### ✔ 2. Flask Web Interface  
Upload an image → view prediction + heatmap instantly.

### ✔ 3. ONNX Export  
Supports ONNX Runtime for optimized inference on CPU/GPU.

---
# 📂 Project Structure
```bash
BrainTumorCNN/
│
├── train.py                 # Model training script
├── dataset.py               # Dataset + transforms
├── model.py                 # CNN model (EfficientNet-B0)
├── export_onnx.py           # Export to ONNX
│
├── requirements.txt         # Core dependencies
│
├── webapp/
│   ├── app.py               # Flask server
│   ├── model_utils.py       # Model loading + Grad-CAM
│   ├── templates/
│   │   └── index.html       # Web UI
│
├── README.md
├── LICENSE
└── .gitignore
```



📦 Dataset
We used the Brain Tumor MRI Dataset from Kaggle:

🔗 https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

Contains 4 categories:

## 🧠 Tumor Classes

| Class      | Description                    |
|------------|--------------------------------|
| Glioma     | Cancerous tumor in glial cells |
| Meningioma | Tumor arising from the meninges |
| Pituitary  | Tumor in the pituitary gland   |
| No Tumor   | Normal brain MRI with no tumor |




---

# 🎯 Model Architecture

### **EfficientNet-B0**  
- Pretrained on ImageNet  
- Input: **224×224** RGB  
- Fine-tuned for 4 tumor classes  
- Adam optimizer  
- Mixed precision training  

### Data Augmentations (Albumentations)
- RandomResizedCrop  
- HorizontalFlip / Rotate  
- ShiftScaleRotate  
- ColorJitter  
- CoarseDropout  
- Normalize  

---

# 🧪 Training the Model

```bash
python train.py \
  --train_dir "Training" \
  --test_dir "Testing" \
  --ckpt_dir "./ckpts" \
  --epochs 30 \
  --batch_size 32
  ```
Best weights are saved to:


Copy code
ckpts/best.pth


⚡ Export to ONNX

```bash
python export_onnx.py \
  --ckpt "./ckpts/best.pth" \
  --out "./ckpts/brain_tumor_model.onnx" \
  --img_size 224
```


## 🌐 Running the Web App

### 📦 Install Dependencies
Run the following commands:

```bash
pip install -r requirements.txt
pip install -r webapp/requirements.txt
pip install opencv-python-headless matplotlib
```
## 📁 Place Your Model Weights

Place your trained model file here:




```bash
ckpts/best.pth
```

After running the server, open:

👉 http://localhost:5000

Upload an MRI image to receive:
- Predicted tumor class  
- Confidence score  
- Grad-CAM heatmap visual explanation  

---

## 📥 Download Model Weights (IMPORTANT)

Model weights are too large to store directly in GitHub.

Download the latest trained weights from:

👉 **https://github.com/Abhijith-Reddy-ch/BrainTumorCNN/releases/latest**


Place inside:
```bash
ckpts/best.pth
```



---

## 🛠 Requirements

### Core Dependencies

torch>=2.0
torchvision
timm
albumentations
numpy
pillow
scikit-learn
opencv-python-headless
tqdm
onnxruntime



### Web App Dependencies

flask==2.2.5
onnxruntime
numpy
Pillow



---

## 🔐 Ethical Use Disclaimer

This model is **not certified**, **not FDA-approved**, and must **not** be used for medical diagnosis.  
It is intended **only for academic and research purposes**.

---

## ⭐ Future Upgrades
- TensorRT acceleration  
- Mobile app version  
- Streamlit / React front-end  
- Grad-CAM++  
- UNet-based tumor segmentation  
- Model quantization for edge devices  

---

## 🙌 Credits
- **Dataset:** Kaggle Brain Tumor MRI Dataset  
- **Backbone Model:** EfficientNet (Tan & Le)  
- **Explainability Method:** Grad-CAM (Selvaraju et al.)  
- **Maintainer:** *Abhijith Reddy Ch & Sai Vikas*  
