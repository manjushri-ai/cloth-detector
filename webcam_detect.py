#!/usr/bin/env python3
"""
webcam_detect.py - live webcam test for the 44-class YOLO26n clothing model.
Read-only use of best.pt: nothing is trained, modified, or saved.

Usage (from the folder containing this script and best.pt):
    python webcam_detect.py
Options:
    --weights best.pt   path to the model (default: best.pt next to this script)
    --conf 0.40         confidence threshold
    --imgsz 512         inference size (512 = what the model was trained at)
    --camera 0          webcam index (try 1 if 0 is the wrong camera)
Press q (with the video window focused) to quit.
"""
import argparse
import os
import sys
import time

import cv2
import torch
from ultralytics import YOLO


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(here, "best.pt"))
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()

    if not os.path.isfile(a.weights):
        sys.exit(f"best.pt not found at: {a.weights}\nPut best.pt next to this script or pass --weights <path>.")

    device = 0 if torch.cuda.is_available() else "cpu"
    half = torch.cuda.is_available()  # FP16 only helps on an NVIDIA GPU
    print(f"device: {'CUDA - ' + torch.cuda.get_device_name(0) if half else 'CPU'}")

    model = YOLO(a.weights)
    print(f"loaded {a.weights} - {len(model.names)} classes")
    if len(model.names) != 44:
        print(f"WARNING: expected 44 classes, model has {len(model.names)} - is this the right best.pt?")

    # CAP_DSHOW opens the webcam much faster on Windows; fall back to the default backend elsewhere
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(a.camera, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open webcam {a.camera}. Close other apps using the camera (Teams/Zoom) or try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win = "YOLO26n 44-class clothing - press q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fps, t_prev = 0.0, time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed - webcam disconnected?")
                break

            results = model.predict(frame, imgsz=a.imgsz, conf=a.conf, device=device,
                                    half=half, max_det=20, verbose=False)
            annotated = results[0].plot()  # boxes + class names + confidence

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))  # smoothed FPS
            t_prev = now
            n = len(results[0].boxes)
            cv2.putText(annotated, f"FPS {fps:4.1f} | detections {n} | conf>={a.conf:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(win, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:  # window closed with the X button
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
