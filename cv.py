import cv2

# --- Konfigurasi ---
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("face-model.yml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

# --- Sesuaikan dengan ID dan Nama Anda ---
names = ['None', 'Burju', 'Eric', 'Antony'] 

# --- Pengaturan Visual ---
confidence_threshold = 70  # Threshold bisa sedikit diturunkan karena model lebih akurat
COLOR_BOX = (0, 255, 0)
COLOR_KNOWN_TEXT = (255, 255, 255)
COLOR_UNKNOWN_TEXT = (0, 0, 255)

# --- Inisialisasi Kamera ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_BOX, 2)
        
        # --- PRE-PROCESSING SEBELUM PREDIKSI (SANGAT PENTING!) ---
        # 1. Crop wajah dari frame grayscale
        face_roi = gray[y:y + h, x:x + w]
        # 2. Normalisasi ukuran ke 200x200 (HARUS SAMA DENGAN DI TRAIN.PY)
        face_resized = cv2.resize(face_roi, (200, 200))
        # 3. Tingkatkan kontras (HARUS SAMA DENGAN DI TRAIN.PY)
        face_equalized = cv2.equalizeHist(face_resized)
        
        # Lakukan prediksi pada gambar yang sudah di-proses
        id, confidence = recognizer.predict(face_equalized)

        # Evaluasi hasil prediksi
        if confidence < confidence_threshold:
            match_percentage = round(100 - confidence)
            display_name = names[id] if id < len(names) else "ID Salah"
            display_text = f"{display_name} ({match_percentage}%)"
            cv2.putText(frame, display_text, (x + 5, y - 5), font, 0.8, COLOR_KNOWN_TEXT, 2)
        else:
            display_text = "Tidak Dikenal"
            cv2.putText(frame, display_text, (x + 5, y - 5), font, 0.8, COLOR_UNKNOWN_TEXT, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord("q"):
        break

print("[INFO] Menutup program.")
cap.release()
cv2.destroyAllWindows()