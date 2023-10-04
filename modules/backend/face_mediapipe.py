import cv2 as cv
import mediapipe as mp
import numpy as np

def plot_image(image:np.ndarray,name_window:str):
    '''
    Plot image
    ----------------------------------------------------------------
    Args:
    image (np.ndarray): image to plot
    name_window (str): window name
    '''
    # show the image, provide window name first
    cv.imshow(name_window, image)
    # add wait key. window waits until user presses a key
    cv.waitKey(0)
    # and finally destroy/close all open windows
    cv.destroyAllWindows()

def One_Face(Result,h,w):
    ''' 
    Function to have a single face
    If there are more than two detections, it selects the two that are closest to the camera
    ----------------------------------------------------------------
    Args:
    - Result: object with result MediaPipe
    - h: height of the image
    - w: width of the image
    '''
    Vec_A = []
    for detection in Result.detections:
        we = int(detection.location_data.relative_bounding_box.width * w)
        hg = int(detection.location_data.relative_bounding_box.height * h)
        Vec_A.append(we*hg)

    Index = Vec_A.index(max(Vec_A))
    return [Result.detections[Index]]

def Norm_Bounding(detection:object):
    '''
    Function to verify that the Bounding box does not exceed the limits of the image.
    In case with points out of the image will remplace with 1.0
    ----------------------------------------------------------------
    Args:
    - detection: object with results of MediaPipe
    Returns:
    - detection: object with results of MediaPipe
    - flags: True if the detection Ok or False if the detection is out of image
    '''
    flag = True
    if detection.location_data.relative_bounding_box.xmin < 0 or detection.location_data.relative_bounding_box.ymin < 0:
        flag = False
    # Verificar x
    if (detection.location_data.relative_bounding_box.xmin + detection.location_data.relative_bounding_box.width
        > 1.0):
        Dif = detection.location_data.relative_bounding_box.xmin + detection.location_data.relative_bounding_box.width - 1
        detection.location_data.relative_bounding_box.width = detection.location_data.relative_bounding_box.width - Dif
        flag = True
    # Verificar y
    if (detection.location_data.relative_bounding_box.ymin + detection.location_data.relative_bounding_box.height
        > 1.0):
        Dif = detection.location_data.relative_bounding_box.ymin + detection.location_data.relative_bounding_box.height - 1
        detection.location_data.relative_bounding_box.height = detection.location_data.relative_bounding_box.height - Dif
        flag = True

    return detection, flag

def face_detect(image:np.ndarray,plot:bool, on_predictions:bool=False):
    '''
    Detect face in image
    ----------------------------------------------------------------
    Args:
    image (np.ndarray): image to detect face in
    plot (bool): if True, plot the image
    on_predictions (bool): if False save predictions, but if true pass to model prediction

    Returns:
    flag (bool): if True, detect face in image
    image_dr : image with points in face
    dic_results: dictionary with information of face detection
    '''
     # Crate mp-face model detection
    mp_face_detection = mp.solutions.face_detection
    # Image original -> image
    if plot:
        plot_image(image,name_window='original image')

    # Image draw results -> image_dw
    ## Flip image and copy
    image_dw = image.copy()
    image_dw = cv.flip(image_dw, 1)
    lb = 'No se detecta rostro'
    flag = False
    img_predict = None

    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.6) as face_detection:

        # Dimnesiones (Relebant for the extraction of stitches)
        height, width, _ = image.shape
        # Convert to RGB, neceesary for MediaPipe
        frame_rgb = cv.cvtColor(image_dw, cv.COLOR_BGR2RGB)
        # Model results
        Result = face_detection.process(frame_rgb)
        if Result.detections is not None:
            lb = 'Se detecto mano'
            if len(Result.detections) > 1:
                # Only one face
                Results = One_Face(Result,height,width)
            else:
                Results = Result.detections
            for detection in Results:
                detection, flag_ok = Norm_Bounding(detection)
                if flag_ok:
                    # Bounding Box
                    xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                    ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                    w = int(detection.location_data.relative_bounding_box.width * width)
                    h = int(detection.location_data.relative_bounding_box.height * height)
                    cv.rectangle(image_dw, (xmin, ymin), (xmin + w, ymin + h), (255, 0, 0), 3)
                    image_dw = cv.flip(image_dw, 1)
                    cv.putText(image_dw, "Score detect face: " + str(round(detection.score[0],3)), [10, 30], cv.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 1)
                    image_dw = cv.flip(image_dw, 1)
                    if on_predictions:
                        flag = True
                        img_predict = image#[ymin:ymin + h,xmin:xmin + w]
                        return flag, image_dw, img_predict, Results
    
    return flag, image_dw, img_predict, None
                


