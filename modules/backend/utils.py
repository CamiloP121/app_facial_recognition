import base64
import numpy as np
import cv2 as cv
import os
import json

def base64toimage(image_data: str, save:bool = True):
    '''
    Converts base64 to image
    ----------------------------------------------------------------
    Arg:
    img_data (str): base64 string
    save (bool, optional): whether to save the image or not. Defaults to True.
    Returns:
    imagae (Opcional)
    '''
    nparr = np.fromstring(base64.b64decode(image_data), np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    if save:
        cv.imwrite('modules/static/temp/image.jpg', image)
        with open('modules/static/temp/image.txt', 'w') as file:
            file.write(image_data)
    else:
        return image