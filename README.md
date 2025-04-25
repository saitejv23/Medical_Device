
# 🧠 Medical Device Detection & Classification API

This project uses two custom-trained YOLO models to detect and classify medical devices from images via a FastAPI interface. The system provides endpoints to upload an image, detect the medical device, and classify the cropped device image.

---

## 📁 Project Structure

```
Checkpoint 3/
├── .env
├── requirements.txt
├── app/
│   └── main.py
├── Models/
│   ├── Detection_model.pt
│   └── Classification_model.pt
├── Datasets&Results/
│   ├── Medical_Device_Classification_Data/
│   └── Medical_Device_Detection_Data/
│   └── ...
├── Notebook files/
│   ├── Medical_device_Classification.ipynb
│   └── Medical_Device_Detection.ipynb
```
---

## ⚙️ Step-by-Step Setup

### 1️⃣ Create a `.env` File

At the root of the project, create a file named `.env` and add the following:

```dotenv
DETECTION_MODEL_PATH=path_to_detection_model
CLASSIFICATION_MODEL_PATH=path_to_classification_model
```

Make sure to replace `path_to_detection_model` and `path_to_classification_model` with the actual paths to your trained YOLO model files.

### 2️⃣ Create Virtual Environment

In your terminal, navigate to the project directory and run the following command to create a virtual environment:

```bash
python -m venv venv
```

This will create a virtual environment named `venv` in your project folder.

### 3️⃣ Activate the Virtual Environment

Activate the virtual environment using the following command:

- **On Windows:**

```bash
venv\Scripts\activate
```

- **On macOS/Linux:**

```bash
source venv/bin/activate
```

### 4️⃣ Install Required Libraries

With the virtual environment activated, install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Uvicorn Server

Once all the required libraries are installed, you can start the FastAPI server with:

```bash
uvicorn app.main:app --reload
```

This will start the server locally. By default, it will be running at `http://127.0.0.1:8000`.

---

## 🚀 API Endpoints

### 1️⃣ Detect Device

- **Endpoint:** `/detect-device/`
- **Method:** `POST`
- **Parameters:** `file` (image)
- **Description:** Detects medical devices from the uploaded image. Returns the label, bounding box coordinates, and a base64-encoded cropped image of the detected device.

### 2️⃣ Classify Device

- **Endpoint:** `/classify-device/`
- **Method:** `POST`
- **Parameters:** `file` (cropped image)
- **Description:** Classifies the medical device from the cropped image. Returns the class name and confidence score.

---

## 🛠️ Tools & Libraries

- **YOLOv5 & YOLOv8** for object detection and classification
- **FastAPI** for building the REST API
- **Pillow (PIL)** for image processing
- **Uvicorn** as the ASGI server
