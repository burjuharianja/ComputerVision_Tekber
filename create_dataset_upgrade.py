import cv2
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
dataset_path = "dataset/"
jumlah_gambar = 200 
min_face_size = (100, 100) 


if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

person_id = input('Masukkan ID Angka untuk orang ini (contoh: 1, 2, atau 3): ')

cap = cv2.VideoCapture(0)
count = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1) # Mirror kamera
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=min_face_size)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        count += 1
        file_path = f"{dataset_path}Person-{person_id}-{count}.jpg"
        cv2.imwrite(file_path, gray[y:y + h, x:x + w])
        
        progress_text = f'Gambar ke-{count}/{jumlah_gambar}'
        cv2.putText(frame, progress_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Pengambilan Dataset", frame)

    k = cv2.waitKey(100) & 0xff 
    if k == ord('q'):
        break
    elif count >= jumlah_gambar:
        break

print(f"\n[INFO] Selesai mengambil {count} gambar.")
cap.release()
cv2.destroyAllWindows()