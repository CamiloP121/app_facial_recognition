from fastapi import FastAPI, Request, File, UploadFile, Form, status, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
#from modules.backend import utils, db
import asyncio
from pathlib import Path
import logging
from starlette.datastructures import URL
#from modules.backend.mediapipe import hands_detect
import cv2
import base64

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

# Predit page
@app.route("/ArkangelAI/predict")
def predict(request:Request):
    title = 'Predicción alfabeto de Señas'
    return templates.TemplateResponse("predict.html", {"request": request, 'title':title})