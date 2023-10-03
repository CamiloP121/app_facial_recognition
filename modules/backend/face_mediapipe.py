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
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    index_list = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
                    122, 196, 3, 51, 281, 248, 419, 351, 37, 0, 267]
    
    # Image original -> image
    if plot:
        plot_image(image,name_window='original image')

     # Image draw results -> image_dw
    ## Flip image and copy
    image_dw = image.copy()
    image_dw = cv.flip(image_dw, 1)
    lb = 'No se detecta rostro'
    flag = False
    dic_result = None

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.65) as face_mesh:

        # Dimnesiones (Relebant for the extraction of stitches)
        height, width, _ = image.shape
        # Convert to RGB, neceesary for MediaPipe
        frame_rgb = cv.cvtColor(image_dw, cv.COLOR_BGR2RGB)
        # Model results
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks is not None:
            lb = 'Se detecto mano'
            # Extract face points
            # dic_result, flag = extract_hand_points(results)
            # Prediction module
            for face_landmarks in results.multi_face_landmarks:
                for index in index_list:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    cv.circle(image_dw, (x, y), 2, (255, 0, 255), 2)
            if on_predictions:
               # image_dw = model_predict(dic_distances,dic_result,image)
               return None, image_dw, None
    
    return flag, image_dw, dic_result
                


