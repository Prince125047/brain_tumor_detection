# Hybrid Quantum-CNN-CBAM Model for Brain Tumor Classification

This repository contains a custom deep learning model that integrates **Convolutional Neural Networks (CNN)**, **Convolutional Block Attention Module (CBAM)**, and a **Quantum Neural Network (QNN)** for classifying brain tumors from MRI scans. The model is built from scratch using TensorFlow and PennyLane, aiming to explore how quantum layers can enhance classical models.

---

## 📂 Dataset

* **Source**: [Brain Tumor MRI Dataset – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* **Classes**: `glioma`, `meningioma`, `pituitary`, `no tumor`
* **Original Training Images**: 5,712
* **After Preprocessing & Augmentation**:

  * Training images: **11,424**
  * Testing images: **1,311**

The dataset was preprocessed (resizing, normalization) and augmented (rotation, flipping, brightness adjustments) using custom scripts provided in the `helper.py` file.

---

##  Model Overview

The proposed model combines classical and quantum layers in a custom architecture:

* **CNN Layers**: Feature extraction using multiple Conv2D and MaxPooling2D blocks.
* **CBAM Block**: Channel and spatial attention to enhance informative features.
* **Quantum Layer**: A PennyLane-based variational quantum circuit using AngleEmbedding and BasicEntanglerLayers.
* **Dense Layers**: Fully connected layers for final classification with softmax output.

---

##  Repository Structure

```
├── model_notebook.ipynb         # Main model architecture, training, evaluation
├── helper.py                    # Preprocessing and augmentation code
├── README.md                    # Project documentation
```

---

##  Training Configuration

* **Input shape**: (256, 256, 3)
* **Loss function**: `sparse_categorical_crossentropy`
* **Optimizer**: Adam
* **Batch size**: 32
* **Epochs**: 15
* **Frameworks**: TensorFlow, PennyLane
* **Hardware**: Trained on Google Colab (T4 GPU)

---

##  How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/hybrid-quantum-cnn-cbam.git
   cd hybrid-quantum-cnn-cbam
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   Open `model_notebook.ipynb` in Jupyter or Google Colab.

---

##  Future Work

* Add Grad-CAM or SHAP-based explainability.
* Explore deeper quantum circuits and more qubits.
* Compare against transformer-based medical models.

---

##  License

This project is for academic and research use only. Please refer to the dataset license on Kaggle before using the data for other purposes.
