from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

model = YOLO("best.pt")


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO26n Clothing Detection</title>
    <style>
        body {
            font-family: Arial;
            text-align: center;
            background: #f5f5f5;
        }
        video, canvas {
            width: 80%;
            max-width: 900px;
            border-radius: 10px;
        }
        #output {
            position: relative;
            display: inline-block;
        }
        canvas {
            position: absolute;
            left: 0;
            top: 0;
        }
        button {
            padding: 12px 25px;
            font-size: 16px;
            margin: 15px;
        }
    </style>
</head>

<body>

<h1>YOLO26n 44-Class Clothing Detection</h1>

<button onclick="startCamera()">Start Camera</button>

<div id="output">
    <video id="video" autoplay playsinline></video>
    <canvas id="canvas"></canvas>
</div>

<script>
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let running = false;

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true
    });

    video.srcObject = stream;
    running = true;

    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        detect();
    };
}

async function detect() {
    if (!running) return;

    const temp = document.createElement("canvas");
    temp.width = video.videoWidth;
    temp.height = video.videoHeight;

    const tempCtx = temp.getContext("2d");
    tempCtx.drawImage(video, 0, 0);

    temp.toBlob(async (blob) => {

        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            for (const d of data.detections) {

                const [x1, y1, x2, y2] = d.bbox;

                const scaleX = canvas.width / data.width;
                const scaleY = canvas.height / data.height;

                const x = x1 * scaleX;
                const y = y1 * scaleY;
                const w = (x2 - x1) * scaleX;
                const h = (y2 - y1) * scaleY;

                ctx.strokeStyle = "#00ff00";
                ctx.lineWidth = 3;
                ctx.strokeRect(x, y, w, h);

                ctx.fillStyle = "#00ff00";
                ctx.font = "18px Arial";

                ctx.fillText(
                    d.class_name + " " + d.confidence,
                    x,
                    Math.max(20, y - 5)
                );
            }

        } catch (error) {
            console.error(error);
        }

        setTimeout(detect, 200);
    }, "image/jpeg", 0.7);
}
</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classes": len(model.names)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    width, height = image.size

    results = model.predict(
        image,
        imgsz=512,
        conf=0.40,
        device="cpu",
        verbose=False
    )

    detections = []

    for box in results[0].boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_name": model.names[class_id],
            "confidence": round(confidence, 2),
            "bbox": [
                round(x1, 1),
                round(y1, 1),
                round(x2, 1),
                round(y2, 1)
            ]
        })

    return {
        "width": width,
        "height": height,
        "detections": detections
    }