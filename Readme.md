# MultiObject Detector

A lightweight custom object detection system built in PyTorch, trained on YOLO-format labeled data and capable of detecting multiple objects per image with bounding box regression and class prediction.

---

## Overview

This project trains a neural network (`MultiObjectDetector`) to detect up to N objects per image simultaneously. It reads images and YOLO-format `.txt` labels, pads detections to a fixed object count, and outputs per-object class scores and bounding boxes. At inference time, predicted boxes are projected back to pixel coordinates and drawn onto the image using OpenCV.

---

## Project Structure

```
project/
├── Banner/                  # Training images (.jpg)
├── obj_Train_data/          # YOLO-format label files (.txt)
├── testing/                 # Single test image for inference
├── object_detector.pth      # Saved model weights (after training)
└── train.py                 # Training + inference script
```

---

## Label Format (YOLO)

Each `.txt` label file contains one row per object:

```
<class_id>  <x_center>  <y_center>  <width>  <height>
```

All values are normalized to `[0, 1]` relative to image dimensions. The label filename must match its image (e.g. `img001.jpg` → `img001.txt`).

---

## Architecture

`MultiObjectDetector` takes a `(batch, 3, 224, 224)` tensor and outputs predictions of shape `(batch, max_objects, num_classes + 4)` — one row per object slot, containing class logits followed by a 4-value bounding box.

> **Note:** The `MultiObjectDetector` class is referenced but not shown in the provided code. It should be defined before the training block and follow the output contract above.

---

## Dataset — `ObjectDataset`

| Detail | Value |
|---|---|
| Input images | `.jpg` files in `img_dir` |
| Labels | Matching `.txt` files in `label_dir` |
| Image resize | 224×224 |
| Max objects per image | 5 (padded with zeros if fewer) |
| Output per sample | `(image_tensor, padded_boxes)` |

Images with no label file return an all-zero box tensor of shape `(max_objects, 5)`.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Batch size | 2 |
| Epochs | 10 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Classification loss | CrossEntropyLoss |
| Bounding box loss | MSELoss |
| Max objects per image | 5 |
| Input resolution | 224×224 |
| Number of classes | 2 *(configurable)* |

---

## Training

```bash
python train.py
```

The training loop iterates over each object slot per prediction, splitting the output into:
- `class_pred` → `preds[:, i, :num_classes]` fed into `CrossEntropyLoss`
- `bbox_pred` → `preds[:, i, num_classes:]` fed into `MSELoss`

Losses are summed across all object slots and all samples before backpropagation. A running total is printed after each epoch.

**Expected output:**
```
Epoch 1 | Loss: 12.4832
Epoch 2 | Loss: 9.7103
...
```

Model weights are saved to `/content/object_detector.pth` after training completes.

---

## Inference

Load the saved model and run on a test image:

```python
model = MultiObjectDetector(num_classes=2)
model.load_state_dict(torch.load("/content/object_detector.pth"))
model.eval()
```

The inference pipeline:

1. Loads and transforms the test image to `(1, 3, 224, 224)`
2. Runs a forward pass to get `(max_objects, num_classes + 4)` predictions
3. Converts YOLO-normalized boxes back to pixel coordinates via `yolo_to_pixel()`
4. Draws bounding boxes and class labels using OpenCV
5. Displays the annotated image in a window

### `yolo_to_pixel`

Converts a YOLO-format box `(_, x_c, y_c, w, h)` — normalized — to absolute pixel corners `(x1, y1, x2, y2)`:

```python
def yolo_to_pixel(box, img_w, img_h):
    _, x, y, w, h = box
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return int(x1), int(y1), int(x2), int(y2)
```

---

## Requirements

```
torch
torchvision
Pillow
opencv-python
```

Install with:

```bash
pip install torch torchvision Pillow opencv-python
```

---

## Known Limitations & Improvement Notes

**Missing model definition.** The `MultiObjectDetector` class must be defined before use. Its output shape must be `(batch, max_objects, num_classes + 4)` for the training loop to work correctly.

**No confidence score.** The current output has no objectness/confidence channel, so every object slot is always drawn at inference — including zero-padded slots. Adding a confidence threshold would suppress empty predictions.

**MSE on bounding boxes.** MSE treats all coordinate errors equally. CIoU or GIoU loss would improve localization, especially for overlapping objects.

**Fixed object count.** `max_objects=5` is a hard ceiling. Images with more detectable objects will be silently truncated during training.

**Single test image path.** The inference block uses a hardcoded path (`/content/testing`). Wrapping this in a function or CLI argument would make the script more reusable.

---

## References

- Redmon, J. et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- OpenCV Python Docs: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
