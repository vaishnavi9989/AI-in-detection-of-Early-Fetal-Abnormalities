# AI-in-detection-of-Early-Fetal-Abnormalities

### 📁 Project Directory Structure
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

⚙ Setup & Installation
