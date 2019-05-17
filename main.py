import cv2
import os

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
video_capture = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('data_train/trainingdata.yml')
 
id=0

def getNbImages(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    max = 0
    for imagePath in imagePaths:      
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        if ID > max:
            max = ID
    return max

i = getNbImages('images/')

name="not found"
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
 
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        print(id)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (11, 57, 215), 2)
        id, conf = rec.predict(gray[y:y+h,x:x+w])
        if id <= 61 or (id >=218 and id <= 255):
            name="Clement"
        elif id >= 62 and id <= 217:
            name="Samixe"
        else:
            name='Inconnu'
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (x, y), font, 1, (11, 57, 215), 2, cv2.LINE_AA)
 
    # Display the resulting frame
    cv2.imshow('Video', frame)
 
    k = cv2.waitKey(1)
    if k == 27 or k == ord('q'):
        imageName = 'No image saved'
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        print(i)
        i+=1
        imageName = 'image_save.{}'.format(i)
        cv2.imwrite('images/' + imageName + ".png", frame)
       
 
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
print(imageName)
 
"""
01 à 40 ==> Samix
41 à 80 ==> Clément
"""