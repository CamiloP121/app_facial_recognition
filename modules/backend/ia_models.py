from keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2 as cv


class Model():
    def __init__(self, path_model_FR:str, dict_labels:dict):
        self.model = load_model(path_model_FR)
        self.threshold = 0.2
        self.labels = dict_labels

    def preprocess_image(self, image):
        image = cv.resize(image,(224,224))
        image = image / 255.0
        image = image.reshape(1, 224, 224, 3)

        return image
    
    def classify(self, prediction):
        prediction /= np.linalg.norm(prediction, ord=2)
        vec_results, vec_labels = [], []
        result = "No Match found"
        score = 0.0
        for name, embed in self.labels.items():
            emb_distance = embed - prediction
            # norm distance 
            embedding_distance_norm = np.linalg.norm(emb_distance)
            tmp = embedding_distance_norm if embedding_distance_norm < self.threshold else 0 # Threshold inverse
            if tmp > 0:
                vec_results.append(tmp)
                vec_labels.append(name)
            
        if vec_results:
            score = min(vec_results)
            result = vec_labels[vec_results.index(score)]

        return result, score
    
    def predict(self, image, img_dw):
        prediction = self.model.predict(image)

        result, score = self.classify(prediction=prediction)
        cv.putText(img_dw, "Score face recognition: " + str(1 - round(score,3)), [10, 60], cv.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 1)
        cv.putText(img_dw, result.upper(), [10, 90], cv.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 1)
        return result


    


