import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model_1= tf.keras.models.load_model("brand_classification_model.h5") # load model for brand classification
model_2= tf.keras.models.load_model("Colddrink_level_classifier_.h5") # load model for cold drink level classification

dir_path='drop_images/' #path to the directory where test images are saved 
def predict(dir_path,model_1,model_2):
    for i in os.listdir(dir_path):
        img=image.load_img(dir_path+ i, target_size=(200,200))
        plt.imshow(img)
        plt.show()
        
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        images=np.vstack([x])
        result_1=model_1.predict(images)
        result_2=model_2.predict(images)
        if result_1==0:
            print('This is coke',end=" ")
        else:
            print('This is pepsi',end=" ")
        if np.argmax(result_2)==0:
            print('100% filled.')
        elif np.argmax(result_2)==1:
            print('less than 50% filled.')
        else:
            print('more than 50% filled.')
        
if __name__ == "__main__":
    predict(dir_path, model_1, model_2)