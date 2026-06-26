# Artificial Vision - Bionic Eye Project

## Folder Structure
```
artificial_vision/
├── main.py                 # Entry point - run this. Opens webcam, detects faces
├── f_Face_info.py           # Face detection + combines age/gender/race/emotion
├── f_main.py                # Face recognition (known/unknown) - placeholder
├── train_cnn.py             # CNN training script (for custom dataset)
├── requirements.txt
├── age_detection/
│   └── f_my_age.py
├── gender_detection/
│   └── f_my_gender.py
├── race_detection/
│   └── f_my_race.py
└── emotion_detection/
    └── f_emotion_de.py
```

## Setup in VS Code

1. Open this folder in VS Code: `File > Open Folder`
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate      (Windows)
   source venv/bin/activate   (Mac/Linux)
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the project:
   ```
   python main.py
   ```
   - This opens your webcam and shows a live window with face boxes +
     age/gender/race/emotion labels.
   - Press `q` to quit.

## Notes
- `f_main.py` (`rec` class) was not detailed in the original project report —
  it's a placeholder here (always returns "unknown"). Plug in real face
  recognition logic (e.g. `face_recognition` library or `DeepFace.find`)
  if you want known-person matching.
- `train_cnn.py` expects a dataset folder structure like:
  ```
  dataset/training/class_a/...
  dataset/training/class_b/...
  dataset/test/class_a/...
  dataset/test/class_b/...
  ```
- First run will download DeepFace's pretrained model weights — needs
  internet access once.
