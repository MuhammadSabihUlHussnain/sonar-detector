# Sonar Object Detector

A static web page that runs a YOLO26s object detector on sonar images, entirely in the
browser. Upload a forward-looking, side-scan or imaging-sonar image and the page draws the
objects it finds, with a confidence slider and the model's measured accuracy for each class,
so you can judge how far to trust a detection. Images never leave your computer.

Built-in samples from the held-out test set show the true labels as dashed boxes to compare against.

## How it works

- `model/sonar-yolo26s.onnx`: YOLO26s (Ultralytics), 416×416 input, exported to ONNX end-to-end
  (output: up to 300 boxes as `x1, y1, x2, y2, score, class`).
- `model/meta.json`: class names plus test-set metrics (mAP50, precision and recall per class).
- `app.js`: letterboxes the image to 416×416, runs the model with
  [onnxruntime-web](https://onnxruntime.ai/) (WebAssembly), maps the boxes back and draws them.
- No build step and no server. Any static host works, GitHub Pages included.

Run it locally with `python -m http.server 8000`, then open http://localhost:8000.
Opening `index.html` straight from disk won't work, because the browser blocks loading the model from `file://`.

## The model

YOLO26s (9.5M parameters) trained on two Kaggle T4 GPUs at 640 px on the full labeled train split
(24.6K images, fish-video frames thinned to every third) of a 65.7K-image sonar dataset. It ran 92
epochs and stopped early; the best epoch (77) is used. The page shows its test scores at 416 px.

| Model | Test mAP50 (416 px) | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| YOLO26s, GPU (current) | 0.800 | 0.503 | 0.837 | 0.804 |
| YOLO26n, CPU demo (previous) | 0.706 | 0.385 | 0.681 | 0.691 |

Both are scored on the same 8,566 held-out test images (28,305 boxes). At 640 px the YOLO26s model
scores 0.811 mAP50.

25 classes: fish, human, plane, ship, boat, tire, bottle, can, cube, ball, cylinder, cage, bucket,
rov, chain, hook, propeller, valve, drink_carton, mine, wall, other_debris, crab_pot, pipeline,
non_mine_object.

## Training data and credits

| Dataset | Sonar | License |
|---|---|---|
| [UATD](https://doi.org/10.6084/m9.figshare.21331143) (Xie, Yang & Qiu, Scientific Data 2022) | Tritech Gemini 1200ik FLS | CC BY 4.0 |
| [Forward-looking sonar detection dataset](https://github.com/XingYZhu/Forward-looking-Sonar-Detection-Dataset) (Nankai) | Oculus M1200d FLS | not stated |
| [Marine Debris FLS](https://github.com/mvaldenegro/marine-debris-fls-datasets) (Valdenegro-Toro et al.) | ARIS Explorer 3000 | CC BY-NC-SA 4.0 |
| [Caltech Fish Counting](https://github.com/visipedia/caltech-fish-counting) (Kay et al., ECCV 2022) | ARIS / DIDSON | MIT |
| [AI4Shipwrecks](https://doi.org/10.7302/dmf4-x492) (U. Michigan) | EdgeTech 2205 side-scan | CC BY 4.0 |
| [SCTD](https://github.com/MingqiangNing/SCTD) | Klein side-scan | not stated |
| [MILCO/NOMBO](https://doi.org/10.6084/m9.figshare.24574879) (Santos et al. 2024) | side-scan | CC BY 4.0 |
| [AquaScan-1K](https://zenodo.org/records/18771165) | side-scan | MIT |
| [GhostVision](https://zenodo.org/records/20056679) (PING Ecosystem) | side-scan | CC BY-SA 4.0 |

Sample images come from UATD, Caltech Fish Counting, AI4Shipwrecks, MILCO, AquaScan and GhostVision
under the licenses above. Part of the training data is non-commercial (Marine Debris FLS), so treat
the model as a research demo, not for commercial use.

Model architecture: [Ultralytics YOLO26](https://docs.ultralytics.com/) (AGPL-3.0).
