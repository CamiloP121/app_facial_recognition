from fastapi import FastAPI, Request, File, UploadFile, Form, status, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from modules.backend import utils
import asyncio
from pathlib import Path
import logging
from starlette.datastructures import URL
from modules.backend.face_mediapipe import face_detect
from modules.backend.ia_models import Model
import cv2
import base64
import json
import pickle
from tensorflow.keras.applications.vgg16 import VGG16

import warnings
warnings.filterwarnings("ignore")

# Crete app
app = FastAPI()
logging.info('Starting App')

# Create directorys necessary
#Path('temp').mkdir(parents=True, exist_ok=True)
Path('logs').mkdir(parents=True, exist_ok=True)
#Path('modules/backend/db').mkdir(parents=True, exist_ok=True)
#Path('modules/backend/models').mkdir(parents=True, exist_ok=True)

# Create database
#db.create_db()

# Configure logging
# Create templates
app.mount("/static", StaticFiles(directory="modules/static"), name="static")
templates = Jinja2Templates(directory="modules/static/templates")

# Create class model
with open('models/matriz_face_detection_v3.pkl', 'rb') as file:
    lables = pickle.load(file)

with open('models/matriz_emotions.pkl', 'rb') as file:
    lables2 = pickle.load(file)

face_recognizer = Model(
    model= VGG16(include_top=True),
    dict_face=lables,
    dict_emotions=lables2
)

# Predit page
@app.route("/ArkangelAI/predict")
def predict(request:Request):
    title = 'Prueba técnica \nCamilo Andrés Pérez Ospino'
    return templates.TemplateResponse("predict.html", {"request": request, 'title':title})

async def capture_video(websocket: WebSocket):
    '''
    Captures video and processes it Mediapipe
    ----------------------------------------------------------------
    Args:
    Websocket: WebSocket (Video text)
    Returns:
    Video procesed
    '''
    while True:
        # Capturar un cuadro de video de la cámara web
        data = await websocket.receive_text()
        # Decodificar los datos de la imagen en base64
        img = utils.base64toimage(data.split(',')[1].encode(), save=False)
        # Procesar el cuadro de video
        flag , img, img_predict, results = face_detect(img, plot=False, on_predictions=True)
        img = cv2.flip(img, 1)
        if flag:
            #[print("||||", detection)  for detection in results]
            img_predict = face_recognizer.preprocess_image(img_predict) 
            result, result_emo = face_recognizer.predict(img_predict, img)
            print("Result: ", result, result_emo)
        _, buffer = cv2.imencode('.jpg', img)
        processed_frame =  base64.b64encode(buffer).decode("ascii")

        # Enviar el cuadro de video procesado a través del socket
        await websocket.send_text(processed_frame)


@app.websocket('/ArkangelAI/predict/ws')
async def websocket_endpoint(websocket: WebSocket):
    # Abrir la conexión del socket
    print('Ws conections...')
    await websocket.accept()
    try:
        # Iniciar la captura de video desde la cámara web y enviar cada cuadro procesado a través del socket
        await capture_video(websocket)
    finally:
        # Cerrar la conexión del socket cuando se cierra la conexión
        await websocket.close()