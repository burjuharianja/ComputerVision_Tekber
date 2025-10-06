import cv2

face_cascade_path = "haarcascade_frontalface_default.xml"
left_eye_cascade_path = "haarcascade_lefteye_2splits.xml"
right_eye_cascade_path = "haarcascade_righteye_2splits.xml"
smile_cascade_path = "haarcascade_smile.xml" 

face_clf = cv2.CascadeClassifier(face_cascade_path)
left_eye_clf = cv2.CascadeClassifier(left_eye_cascade_path)
right_eye_clf = cv2.CascadeClassifier(right_eye_cascade_path)
smile_clf = cv2.CascadeClassifier(smile_cascade_path)

if face_clf.empty() or left_eye_clf.empty() or right_eye_clf.empty() or smile_clf.empty():
    print("Error: Gagal memuat satu atau lebih file cascade XML.")
    exit()

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Kamera tidak dapat diakses.")
    exit()

while True:
    _, frame = camera.read()
    # mirror kamrea
    frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.equalizeHist(gray)

    faces = face_clf.detectMultiScale(gray_equalized, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Loop untuk setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Buat Region of Interest (ROI) dari area wajah
        roi_gray = gray_equalized[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # 2. Deteksi MATA KIRI
        left_eyes = left_eye_clf.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        for (ex, ey, ew, eh) in left_eyes:
            # Gambar persegi panjang hijau di sekitar mata kiri
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
        # 3. Deteksi MATA KANAN
        right_eyes = right_eye_clf.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        for (ex, ey, ew, eh) in right_eyes:
            # Gambar persegi panjang ungu di sekitar mata kanan
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 255), 2)
            
        # 4. Deteksi SENYUMAN
        smiles = smile_clf.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=28)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            cv2.putText(frame, "Smile!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Tugas Tekber Deteksi Wajah", frame)

    
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()