import cv2
import os
import numpy as np
from PIL import Image
 
recognizer = cv2.face.LBPHFaceRecognizer_create()
 
path = "images"
def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:      
    # Read the image and convert to grayscale
        facesImg = Image.open(imagePath).convert('L')
        faceNP = np.array(facesImg, 'uint8')
        # Get the label of the image
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        # Detect the face in the image
        faces.append(faceNP)
        IDs.append(ID)
        cv2.imshow("Adding faces for training", faceNP)
        cv2.waitKey(10)
    return np.array(IDs), faces
 
Ids, faces = getImagesWithID(path)
recognizer.train(faces, Ids)
recognizer.save("data_train/trainingdata.yml")