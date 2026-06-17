# Marine Life Computer Vision System

A multi-task deep learning pipeline for underwater marine life analysis, built using multi-stage computer vision pipelines.

---

## What it does

Three computer vision tasks on a marine-life dataset containing five species: **fish, jellyfish, shark, tuna, and whale**.

| Task | Description | Model | Result |
|------|-------------|-------|--------|
| **Classification** | Identify the species in an image | Xception (Transfer Learning) | **97.4% accuracy** |
| **Detection** | Locate and draw bounding boxes around animals | YOLOv8n | **mAP@0.5: 74.4%** |
| **Few-Shot Recognition** | Identify individual whales by identity | Siamese Network + Triplet Loss | **Top-5: 91.7%** |

---

## Models

- **Xception** — pretrained ImageNet backbone fine-tuned for 5-class species classification
- **YOLOv8n** — real-time object detector trained on 4 species (fish, jellyfish, shark, tuna)
- **Siamese Network** — metric learning model with triplet loss for individual whale re-identification using only a few reference photos per whale

---

## Project Structure

```
├── Marine_Life_CV_Project.ipynb   # Main notebook (all 3 tasks)
├── requirements.txt
├── README.md
└── dataset/                       # Not included — see Dataset section below
    ├── classification/
    │   ├── train/<class>/*.jpg
    │   └── val/<class>/*.jpg
    ├── detection/
    │   ├── train/<class>/{*.jpg, *.txt}
    │   └── val/<class>/{*.jpg, *.txt}
    └── few-shot-learning/
        ├── train/whale/<whale_id>/*.jpg
        └── val/whale/<whale_id>/*.jpg
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/marine-life-cv.git
cd marine-life-cv
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your dataset**

Place your dataset following the folder structure shown above, then update `DATA_ROOT` in the configuration cell of the notebook:
```python
DATA_ROOT = "/path/to/your/dataset"
```

**4. Run the notebook**

Open `Marine_Life_CV_Project.ipynb` in Jupyter and run the cells in order.

> **Note:** Training is computationally intensive. A GPU is strongly recommended (Google Colab with T4 works well). Pre-trained weights are saved after each task so you only need to train once — subsequent runs can reload saved models directly.

---

## Saved Model Files

After training, three model files are saved:

| File | Task |
|------|------|
| `marine_classifier.h5` | Species classifier |
| `marine_detector/weights/best.pt` | YOLOv8 detector |
| `whale_encoder.weights.h5` | Siamese encoder weights |


---

## Technologies

`Python` `TensorFlow/Keras` `YOLOv8` `OpenCV` `scikit-learn` `NumPy` `Matplotlib`
