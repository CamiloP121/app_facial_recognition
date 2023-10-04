from keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2 as cv


class Model():
    def __init__(self, model, dict_face:dict,dict_emotions:dict):
        self.model = model
        self.threshold_face = 0.2
        self.threshold_emo = 0.8
        self.labels_face = dict_face
        self.labels_emo = dict_emotions

    def preprocess_image(self, image):
        image = cv.resize(image,(224,224))
        image = image / 255.0
        image = image.reshape(1, 224, 224, 3)

        return image
    
    def classify(self, prediction, threshold, labels, defalult):
        vec_results, vec_labels = [], []
        result = defalult
        score = 0.0
        for name, embed in labels.items():
            emb_distance = embed - prediction
            # norm distance 
            embedding_distance_norm = np.linalg.norm(emb_distance)
            tmp = embedding_distance_norm if embedding_distance_norm < threshold else 0 # Threshold inverse
            if tmp > 0:
                vec_results.append(tmp)
                vec_labels.append(name)
            
        if vec_results:
            score = min(vec_results)
            result = vec_labels[vec_results.index(score)]

        return result, score
    
    def predict(self, image, img_dw):
        prediction = self.model.predict(image)
        prediction /= np.linalg.norm(prediction, ord=2)
        # Face detection
        result_face, score_face = self.classify(prediction=prediction, threshold=self.threshold_face, 
                                      labels=self.labels_face, defalult="No Match found")
        cv.putText(img_dw, "Score face recognition: " + str(1 - round(score_face,3)), [10, 60], cv.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 1)
        cv.putText(img_dw, result_face.upper(), [10, 90], cv.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 1)

        # Emotions
        result_emo, score_emo = self.classify(prediction=prediction, threshold=self.threshold_emo, 
                                      labels=self.labels_emo, defalult="nuetral")

        cv.putText(img_dw, "Score emotion \nrecognition: " + str(1 - round(score_emo,3)), [10, 120], cv.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 1)
        cv.putText(img_dw, result_emo.upper(), [10, 150], cv.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 1)

        return result_face, result_emo


    


