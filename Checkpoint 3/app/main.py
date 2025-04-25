import os
import io
import base64
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

DETECTION_MODEL_PATH = os.getenv("DETECTION_MODEL_PATH")
CLASSIFICATION_MODEL_PATH = os.getenv("CLASSIFICATION_MODEL_PATH")

# Load YOLO models
try:
    detection_model = YOLO(DETECTION_MODEL_PATH)
    classification_model = YOLO(CLASSIFICATION_MODEL_PATH)
    logger.info("YOLO models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise RuntimeError("Failed to load YOLO models")

app = FastAPI()

# Constants
TEMP_INPUT_PATH = "temp_input.jpg"
TEMP_CROPPED_PATH = "temp_cropped_image.jpg"

@app.post("/detect-device/")
async def detect_device(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        image.save(TEMP_INPUT_PATH)

        results = detection_model.predict(source=TEMP_INPUT_PATH, conf=0.3)
        os.remove(TEMP_INPUT_PATH)

        result = results[0]
        if not result.boxes:
            raise HTTPException(status_code=404, detail="No device detected")

        box = result.boxes[0]
        cls_id = int(box.cls[0])
        label = result.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        bbox = [x1, y1, x2, y2]

        cropped = image.crop((x1, y1, x2, y2))
        buffered = io.BytesIO()
        cropped.save(buffered, format="JPEG")
        cropped_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "label": label,
            "bbox": bbox,
            "cropped_image_base64": cropped_base64
        }

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    except Exception as e:
        logger.exception("Error in /detect-device/")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/classify-device/")
async def classify_device(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        image.save(TEMP_CROPPED_PATH)

        results = classification_model.predict(source=TEMP_CROPPED_PATH, conf=0.3)
        os.remove(TEMP_CROPPED_PATH)

        top_class_index = int(results[0].probs.top1)
        top_class_confidence = float(results[0].probs.top1conf)

        class_name = classification_model.names[top_class_index]

        return JSONResponse(content={
            "label": class_name,
            "confidence": round(top_class_confidence, 2)
        })

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    except Exception as e:
        logger.exception("Error in /classify-device/")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
