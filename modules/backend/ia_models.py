from keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2 as cv
import json


class Model():
    def __init__(self, path_model_FR:str, dict_labels:dict):
        self.model = load_model(path_model_FR)
        self.threshold = 0.6
        self.labels = dict_labels

    def preprocess_image(self, image):
        image = cv.resize(image,(224,224))
        image = image / 255.0
        image = image.reshape(1, 224, 224, 3)

        return image
    
    def classify(self, prediction):
        print(prediction)
        max_index = np.argmax(prediction)
        print(max_index, self.labels[str(max_index)])
        print("111111111111111")
        indices_above_threshold = np.where(prediction > self.threshold)
        if indices_above_threshold[1].size > 0:
            max_index = np.argmax(prediction)
            print(max_index, self.labels[str(max_index)])
            result = self.labels[str(max_index)]
        else:
            result = "unknown"
        print(result)

        return result
    def predict(self, image):
        prediction = self.model.predict(image)

        result = self.classify(prediction=prediction)
        return result


    


