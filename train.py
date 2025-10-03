import cv2
import numpy as np
import os

dataset_path = "dataset/"
model_path = "face-model.yml"
cascade_path = "haarcascade_frontalface_default.xml" 

face_detector = cv2.CascadeClassifier(cascade_path)
if face_detector.empty():
    print("[ERROR] Gagal memuat file Haar Cascade.")
    exit()

def get_images_and_labels(path):
    # ... (isi fungsi ini tidak berubah, sudah optimal)
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples, ids, skipped_files = [], [], 0
    print("[INFO] Memindai gambar di dataset...")
    for image_path in image_paths:
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        filename = os.path.basename(image_path)
        try:
            current_id = int(filename.split('-')[1])
        except (ValueError, IndexError):
            print(f"Peringatan: Melewatkan file format salah: {filename}")
            skipped_files += 1
            continue
        faces = face_detector.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) != 1:
            print(f"Peringatan: Ditemukan {len(faces)} wajah di {filename}. Dilewati.")
            skipped_files += 1
            continue
        for (x, y, w, h) in faces:
            face_roi = img_gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (200, 200))
            face_equalized = cv2.equalizeHist(face_resized)
            face_samples.append(face_equalized)
            ids.append(current_id)
    print(f"[INFO] Data disiapkan. Total file dilewati: {skipped_files}")
    return face_samples, np.array(ids)

# --- Menggunakan Parameter LBPH yang sudah dioptimalkan ---
recognizer = cv2.face.LBPHFaceRecognizer.create(radius=1, neighbors=8, grid_x=8, grid_y=8)

if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
    print(f"[ERROR] Folder '{dataset_path}' tidak ada atau kosong.")
else:
    print("\n[INFO] Memulai training model...")
    faces, ids = get_images_and_labels(dataset_path)
    if len(faces) > 0:
        recognizer.train(faces, ids)
        recognizer.write(model_path)
        print(f"\n[SUCCESS] Training selesai. Model disimpan sebagai '{model_path}'")
    else:
        print("[ERROR] Tidak ada data valid untuk ditraining.")