import cv2

faceCascade = cv2.CascadeClassifier('C:\\Users\\Samix-PC-Bureautique\\Desktop\\Cours\\Programmation_temps_reel\\opencv\\sources\\data\\haarcascades_cuda\\haarcascade_frontalface_alt2.xml')
video_capture = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('C:\\Users\\Samix-PC-Bureautique\\Desktop\\Cours\\Programmation_temps_reel\\Code\\data_train\\trainingdata.yml')
# Attention au FPS
video_capture.set(cv2.CAP_PROP_FPS , 20)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

id=0
i=0
name="None"
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # print(id)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (11, 57, 215), 2)
        id, conf = rec.predict(frame[y:y+h,x:x+w])
        if (id >= 0 and id <= 23):
            name = 'Clement'
        if (id >= 23 and id <= 81):
            name = 'Samix'
        # print(id)
        print(name)
        fps = '{} FPS'.format(video_capture.get(cv2.CAP_PROP_FPS))
        print(fps)
        # print('-----')

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (x, y), font, 1, (11, 57, 215), 2, cv2.LINE_AA)

    # Display the resulting frame
    # cv2.imshow('Video', frame)

    k = cv2.waitKey(1)
    if k == 27 or k == ord('q'):
        imageName = 'No image saved'
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        print(i)
        i+=1
        imageName = 'image.{}'.format(i)
        cv2.imwrite('image/' + imageName + ".png", frame)
        

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
print(imageName)