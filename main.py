from model import make_prediction
from fastapi import FastAPI, UploadFile
from PIL import Image

app = FastAPI()


@app.get("/")
def index():
    return {"message": "FastAPI Vision-and-Language Transformer app"}


@app.post("/predict")
def predict(image: UploadFile, text: str):
    image = Image.open(image.file)
    prediction = make_prediction(image=image, text=text)
    return {"Predicted answer": prediction}

