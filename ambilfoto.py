import cv2
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

person_name = input('\nMasukkan nama orang: ')
person_id = input('Masukkan ID untuk orang ini (contoh: 1, 2, atau 3): ')

count = 0
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1) # mirror kamera  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.2, 
                                        minNeighbors=5, minSize=(30, 30))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        count+=1
        file_path = f"{dataset_path}Person-{person_id}-{count}.jpg"
        cv2.imwrite(file_path, gray[y:y + h, x:x + w])
        cv2.putText(frame, f'Mengambil gambar ke-{count}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


    cv2.imshow("Camera", frame)

    k = cv2.waitKey(100) & 0xff # Menunggu 100ms, memberi waktu untuk menyimpan gambar
    if k == ord('q'): 
        break
    elif count >= 30: 
        break


cap.release()
cv2.destroyAllWindows()