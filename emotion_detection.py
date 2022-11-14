import cv2
from deepface import DeepFace


global delay # timer om de deepface analyse maar op de zoveel keren van de while loop uit te laten voeren
            # bv door maar op de 50 keer door de while te gaan een analyse maken, om geheugen te besparen
delay = 0

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'  #getting a haarcascade xml file
face_cascade = cv2.CascadeClassifier()  #processing it for our project


def emotion_detection():

    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  #adding a fallback event
        print("Error loading xml file")

    video = cv2.VideoCapture(0)  # parameter 0 omdat we maar 1 camera hebben (1 webcam), indien bv. 2 camera's -> parameter 1, enz ...

    # Check if the webcam is opened correctly
    if not video.isOpened():
        raise IOError("Cannot open webcam")

    while video.isOpened():  #checking if are getting video feed and using it
        isFrameCaptured, frame = video.read()  #  ret is a Boolean value returned by the read function, and it indicates whether or not the frame was captured successfully. If the frame is captured correctly, it's stored in the variable frame.

        if not isFrameCaptured:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        faces = detect_face(frame)
        result = check_emotion_every_X_seconds(frame, delay)
        

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)  # vierkant rond gezicht zetten + kleur
            
            # emotie op webcam beeld afdrukken
        cv2.putText(frame,
            result,
            (0, 60-10), # coördinaten tekst emotie
            cv2.FONT_HERSHEY_SIMPLEX, 1, # font instellen (Hershey Simplex)
            (0, 0, 255),
            2,
            cv2.LINE_4)

        #this is the part where we display the output to the user
        cv2.imshow('video', frame)

        if cv2.waitKey(1) == ord('q'):  # klik op 'q' toets op af te sluiten
            break

    video.release()
    cv2.destroyAllWindows()


def check_emotion(frame):
    #making a try and except condition in case of any errors
    try:
        analyze = DeepFace.analyze(frame, actions=['emotion'])
        result = analyze['dominant_emotion']
    except:
        result = "no face"

    return result

    
def detect_face(frame):
    
    gray = cv2.cvtColor(frame,
                        cv2.COLOR_BGR2GRAY)  #changing the video to grayscale to make the face analisis work properly
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    return faces

def check_emotion_every_X_seconds(frame, timer):
    if timer % 50 == 0:  # voer analyse maar uit op de 10 keer (1 sec) door de while te gaan
        result = check_emotion(frame)
    
    if (timer < 500):
        timer += 1  # timer incrementeren
    else:
        timer = 0  # timer resetten, zodat deze niet te groot wordt

    return result


if __name__ == '__main__':
    emotion_detection()