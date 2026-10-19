# 🌱 **EcoSort — Waste Segregation Using Edge AI**

### **Intel SIG Recruitment Challenge | Edge AI Track**

**Developer:** Kinjal Barman  
**Academic Status:** 1st-Year B.Tech Student  
**Domain:** Artificial Intelligence, Computer Vision & Sustainability

---

## 📌 **About the Project**

EcoSort is a **CNN-based image classification project** that classifies waste into six categories: **cardboard, glass, metal, paper, plastic, and trash**. It aims to support smarter waste segregation using lightweight Edge AI.

## 🛠️ **Technologies Used**

- **Python** — Programming language
- **PyTorch** — Deep learning and model training
- **Torchvision** — Image preprocessing and augmentation
- **ONNX** — Model export and interoperability
- **ONNX Runtime** — Model inference
- **NumPy** — Numerical computations
- **Pillow** — Image processing
- **tqdm** — Progress bars
- **Matplotlib** — Data visualization
- **Scikit-learn** — Machine learning utilities and evaluation

## 📂 **Dataset**

The dataset contains **1,140 images**:

- **Training:** 900 images
- **Validation:** 240 images

**[Download the EcoSort Dataset](https://drive.google.com/file/d/12lDsBBYZHPM_a8vV2D4CRgxgPH0VI-ZZ/view?usp=sharing)**

## 🚀 **How to Run**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Prepare the Dataset**
```bash
python download_data.py
```

### **3. Train the Model**
```bash
python train_template.py --epochs 10 --batch_size 32 --lr 0.001
```

### **4. Verify the Model**
```bash
python verify_submission.py --model model.onnx --val_dir data/public/val
```

## 🎯 **Challenge Requirements**

- **Target Accuracy:** At least 75%
- **Maximum Model Size:** 1.50 MB
- **Maximum Parameters:** 350,000
- **Input Shape:** `(Batch, 3, 96, 96)`
- **Output Shape:** `(Batch, 6)`
- **Output Format:** ONNX (`model.onnx`)

## 🌍 **Objective**

To explore how lightweight AI models can improve **waste segregation, recycling efficiency, and environmental sustainability**.

---

**Developed by Kinjal Barman | First-Year B.Tech Student**

