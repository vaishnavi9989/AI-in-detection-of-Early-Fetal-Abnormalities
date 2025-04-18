# AI in detection of early Fetal Abnormalities


AI system for early detection of fetal abnormalities using a multi-modal approach that combines ultrasound images and non-invasive ECG signals to identify brain malformations (CSP, LV) and cardiac arrhythmias through real-time, automated predictions.
Traditional fetal anomaly screening methods could be reliant on subjective interpretation of ultrasound scans and fetal heart rhythm monitoring, with limitations due to subjectivity, operator variability, and detection at late gestational ages. Our approach differs by combining the application of deep learning algorithms in order to enable automation and enhanced diagnostic accuracy with real-time and non-invasive assessment of abnormalities in brain structure as well as in cardiac rhythm abnormalities.
---

### 📁 Project Directory Structure
```
text
📂 ECG/  
│   ├── ECG_Model.ipynb                 # ECG model notebook  
│   ├── ecg_cnn_lstm_final.pth          # Trained ECG model  
│   └── Dataset2_NonInvasive/  
│       ├── set-a/  
│       │   └── set-a/                  # Raw ECG data (.hea, .dat, .fqrs, .fqrs.txt, .csv)  
│       └── set-b/  
│           └── set-b/                  # Raw ECG data (.hea, .dat, .csv)  

📂 Ultrasound/  
│   ├── Ultrasound_Model.ipynb          # Ultrasound model notebook  
│   ├── ultrasound_resnet_model.pth     # Trained ResNet18 model  
│   ├── test/                           # Test set for evaluation  
│   │   ├── *.png                       # Test ultrasound images  
│   │   └── _classes.csv                # Labels for test images  
│   ├── train/                          # Training set  
│   │   ├── *.png                       # Train ultrasound images  
│   │   └── _classes.csv                # Labels for training images  
│   ├── valid/                          # Validation set  
│   │   ├── *.png                       # Validation ultrasound images  
│   │   └── _classes.csv                # Labels for validation images  
│   └── Dataset2/                       # Unlabeled fetal head images (generalization testing)  

📄 Fusion_MODEL.ipynb                   # Fusion model notebook (ECG + Ultrasound)
📄 fusion_model.pth                     # Trained fusion model weights

📂 reports/                             # Grad-CAM visualizations and final prediction results
```
⚙ Setup & Installation

### 1. Clone the repository
git clone [https://github.com/VKasarla05/multi-modal-fetal-abnormality-detection-ai.git]


cd multi-modal-fetal-abnormality-detection-ai

### 2. Install required packages


pip install -r requirements.txt


---

## 🚀 How to Run the Models

This section describes how to train and evaluate each model in the project: *Ultrasound, **ECG, and **Fusion*. All notebooks are designed to be run independently and are well-commented for step-by-step execution.

---

### 🧠 1. Ultrasound Model  
📍 Path: Ultrasound/Ultrasound_Model.ipynb

- Uses labeled ultrasound images to detect brain abnormalities: *CSP* and *LV*.
- The dataset is split into:
  - train/ – used for model training  
  - valid/ – used for validation during training  
  - test/ – used for final evaluation  
  - Each folder includes an accompanying _classes.csv file with image labels.
- Trains a *ResNet18* model from torchvision for multi-label classification.
- After training, the model is saved as ultrasound_resnet_model.pth.
- Evaluates model performance using accuracy, F1-score, and Grad-CAM visualizations.
- Generalization is tested on *unlabeled Dataset2*, representing unseen fetal head images.

---

### 💓 2. ECG Model  
📍 Path: ECG/ECG_Model.ipynb

- Processes *4-channel, 10-second fetal ECG signals* using a *1D CNN + Bi-LSTM* architecture.
- Performs filtering, normalization, and segmentation of ECG data from PhysioNet’s set-a and set-b.
- Uses *oversampling* to balance normal and abnormal classes.
- Trains and evaluates the model on non-invasive fetal ECG data.
- Saves the model as ecg_cnn_lstm_final.pth.
- Outputs classification results (Normal/Abnormal) and waveform-based confidence.

---

### 🔗 3. Fusion Model  
📍 Path: Fusion_MODEL.ipynb

- Combines outputs from the Ultrasound and ECG models:
  - CSP & LV abnormality scores
  - ECG normal/abnormal probabilities
- The *Fusion Neural Network* is a lightweight 2-layer FC classifier trained to predict overall fetal health status.
- Supports:
  - Ultrasound-only inference  
  - ECG-only inference  
  - Full fusion mode (both modalities)
- Evaluates performance and generates visual diagnostic reports stored in /reports/ folder.
- Final model is saved as fusion_model.pth.

---

## 📊 Model Performance Summary

| Model               | Accuracy       | F1 Score        | Recall     | Notable Strengths                         |
|---------------------|----------------|-----------------|------------|--------------------------------------------|
| Ultrasound (CSP)    | 84.3%          | 0.8175          |    78%     | Detects midline brain structures (CSP)     |
| Ultrasound (LV)     | 85.3%          | 0.7805          |    72%     | Detects lateral ventricle abnormalities    |
| ECG                 | 98.2%          | 0.9827          |    98%     | Captures waveform + rhythm perfectly       |
| Fusion Model        | 96.39%         | 0.9719          |    97%     | Balanced and robust across modalities      |

> 📌 All metrics evaluated on the test sets after training. Grad-CAM and waveform-based explainability included.

---
## 🎥 Project Presentation (YouTube)

Watch our final project presentation here:

[![Watch the video](https://img.youtube.com/vi/qOPBxL6EJgI/0.jpg)](https://youtu.be/qOPBxL6EJgI)

> 📺 Click the thumbnail above or [watch on YouTube](https://youtu.be/qOPBxL6EJgI)

---

## 📥 Dataset Links

- *Ultrasound Dataset* (CSP & LV labeled):  
  https://universe.roboflow.com/eya-ben-moulehem-o10rs/fetal-brain-ultrasound-fdoq1/dataset/2

- *Ultrasound Dataset for Generalization testing*:  
  https://www.kaggle.com/datasets/nirmalgaud/diverese-fetal-head-images
  
- *ECG Dataset* (NI-FECG, PhysioNet):  
  https://physionet.org/content/challenge-2013/1.0.0/

---

## 🧪 Sample Reports

Sample visual outputs generated by the model:

- Test_ECG_A01_report.png — ECG classification and waveform visual  
- Test_US_01_report.png — Ultrasound classification with Grad-CAM  
- Test_Fusion_report.png — Final fusion diagnosis  
- Grad-CAM overlays and confidence bars saved in /reports/

---

## 📜 License

This project is licensed under the *MIT License*.  
You are free to use, modify, and distribute this work with attribution.

---

## 👩‍💻 Authors

- *Vyshnavi Priya Kasarla*  
- *Vaishnavi Perka*

Contributions, feedback, and forks are welcome!

---
