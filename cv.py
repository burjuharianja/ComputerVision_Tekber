import cv2

recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("face-model.yml")
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

# ID 1 -> Burju, ID 2 -> Eric, ID 3 -> Antony, dst.
names = ['None', 'Burju', 'Eric', 'Antony'] 

confidence_threshold = 75 

COLOR_KNOWN = (255, 255, 255)
COLOR_BOX = (0, 255, 0)    
COLOR_UNKNOWN = (0, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Loop untuk setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_BOX, 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < confidence_threshold:
            match_percentage = round(100 - confidence)
            
            if id < len(names):
                display_name = names[id]
                display_text = f"{display_name} ({match_percentage}%)"
            else:
                display_text = "ID Salah" 
            
            cv2.putText(frame, display_text, (x + 5, y - 5), font, 1, COLOR_KNOWN, 2)
        else:
            display_text = "Tidak Dikenal"
            cv2.putText(frame, display_text, (x + 5, y - 5), font, 1, COLOR_UNKNOWN, 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()