<div align="center">

# 👕 Cloth Detector

**Real-time clothing and accessory detection powered by YOLO26n**

Detects 44 clothing and accessory classes from a live webcam feed, served through a FastAPI backend with a browser-based interface.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started (Docker)](#getting-started-docker)
- [Using the Web App](#using-the-web-app)
- [Alternative: Native Python Webcam](#alternative-native-python-webcam)
- [Supported Garments](#supported-garments)
- [Model Limitations](#model-limitations)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

Cloth Detector is an object detection system built on **YOLO26n** that identifies clothing and accessories in real time. It ships as a self-contained Docker image, so it runs on any machine with Docker installed — no Python environment setup required.

**Key features:**
- 🎯 44-class clothing and accessory detection
- ⚡ Real-time inference on live webcam feed
- 🌐 Browser-based UI — no installation beyond Docker
- 🐳 One-command Docker deployment
- 🐍 Optional native OpenCV script for local Python use

---

## Model Performance

| Metric | Value |
|---|---|
| Model | YOLO26n |
| Task | Object Detection |
| Classes | 44 |
| Validation Images | 26,208 |
| mAP@50 | **75.8%** |
| mAP@50-95 | **71.6%** |
| Weights File | `best.pt` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Detection Model | YOLO26n (Ultralytics) |
| Backend | FastAPI |
| Server | Uvicorn |
| Containerization | Docker |
| Native CV | OpenCV |

---

## Project Structure

```text
cloth-detector/
│
├── best.pt              # Trained YOLO26n weights (44 classes)
├── app.py                # FastAPI backend + browser webcam interface
├── webcam_detect.py       # Standalone OpenCV webcam script
├── Dockerfile             # Docker image definition
├── requirements.txt       # Python dependencies
├── .dockerignore
└── README.md
```

---

## Getting Started (Docker)

Docker is the recommended way to run this project — it requires no local Python setup.

### Prerequisites

| Requirement | Purpose |
|---|---|
| [Git](https://git-scm.com/) | Clone the repository |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Build and run the container |
| Chrome or Edge | Access the web app |
| Webcam | Live detection input |

Verify your installation:

```bash
git --version
docker --version
```

Make sure **Docker Desktop is running** before proceeding.

### 1. Clone the repository

```bash
git clone https://github.com/manjushri-ai/cloth-detector.git
cd cloth-detector
```

### 2. Build the Docker image

```bash
docker build -t cloth-detector .
```

> First build may take a few minutes while dependencies install and the model is copied into the image.

### 3. Run the container

```bash
docker run --rm -p 8000:8000 cloth-detector
```

Keep this terminal open. You should see:

```text
Uvicorn running on http://0.0.0.0:8000
```

### 4. Stop the application

In the terminal running the container, press `Ctrl + C`. Since the container was started with `--rm`, it's automatically removed on stop.

---

## Using the Web App

1. Open **Chrome** or **Edge** and navigate to:

   ```text
   http://localhost:8000
   ```

   > ⚠️ Use `http://localhost:8000`, **not** `http://0.0.0.0:8000` — `0.0.0.0` is not a valid client-side address.

2. Click **Start Camera** and allow camera access when prompted.
3. Select a webcam if multiple are available.
4. Detections appear live with bounding boxes, class labels, and confidence scores:

   ```text
   Tank Top/Undershirt   0.81
   Shorts/Skort          0.90
   Socks                 0.86
   ```

---

## Alternative: Native Python Webcam

For testing without Docker, use the standalone `webcam_detect.py` script.

```bash
python --version                      # Requires Python 3
pip install -r requirements.txt
pip install opencv-python
python webcam_detect.py
```

A window will open showing live detections. Press `q` to exit.

---

## Supported Garments

The model supports 44 clothing and accessory classes.

<details>
<summary><strong>View all 44 classes</strong></summary>

| # | Class | # | Class |
|---|---|---|---|
| 1 | Shirt/Blouse | 23 | Trench Coat |
| 2 | T-Shirt | 24 | Raincoat/Windbreaker |
| 3 | Polo Shirt | 25 | Puffer/Parka |
| 4 | Tank Top/Undershirt | 26 | Bomber/Varsity/Track Jacket |
| 5 | Sweater/Jumper | 27 | Kurta |
| 6 | Cardigan | 28 | Saree |
| 7 | Vest/Waistcoat | 29 | Sherwani |
| 8 | Hoodie | 30 | Salwar Kameez |
| 9 | Sweatshirt | 31 | Lehenga |
| 10 | Swimwear | 32 | Necktie |
| 11 | Trousers/Pants | 33 | Scarf/Shawl |
| 12 | Jeans | 34 | Gloves |
| 13 | Sweatpants/Track Pants | 35 | Hat/Cap |
| 14 | Leggings | 36 | Socks |
| 15 | Shorts/Skort | 37 | Belt |
| 16 | Capri/Cropped Trousers | 38 | Underwear/Briefs |
| 17 | Skirt | 39 | Bra |
| 18 | Dress | 40 | Pyjama Set |
| 19 | Gown | 41 | Bathrobe/Robe |
| 20 | Jumpsuit/Overalls | 42 | Suspenders |
| 21 | Blazer/Suit Jacket | 43 | Footwear |
| 22 | Coat | 44 | Bag/Handbag |

</details>

---

## Model Limitations

Performance varies across garment categories.

### Classes with no training examples

The following classes had **no training examples** in the current training dataset and are therefore **not expected to detect reliably**:

- Polo Shirt
- Hoodie
- Trench Coat
- Puffer/Parka
- Bomber/Varsity/Track Jacket

### Factors affecting detection quality

Detection quality may also vary depending on:

- Garment type
- Lighting conditions
- Camera angle
- Garment visibility (partial occlusion, layering)
- Background clutter
- Similar-looking clothing categories

> These limitations reflect the current training dataset and are expected to improve with additional labeled data for the affected classes.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `docker: command not found` | Install Docker Desktop and ensure it's running |
| Browser can't reach `localhost:8000` | Confirm the container is running and port `8000` isn't in use by another process |
| Camera permission denied | Check browser site settings and allow camera access for `localhost` |
| No webcam detected | Verify OS-level camera permissions and that no other app is using the webcam |
| Slow first build | Normal — dependencies are being installed and the model copied into the image |

---

## Roadmap

- [ ] Add REST API documentation (Swagger/OpenAPI)
- [ ] Support image/video file uploads in addition to live webcam
- [ ] Add unit tests and CI pipeline
- [ ] Publish prebuilt image to Docker Hub

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Built with YOLO26n • FastAPI • Docker

</div>
