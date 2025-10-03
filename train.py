import cv2
import numpy as np
import os


dataset_path = "dataset/"
model_path = "face-model.yml"

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    
    face_samples = []
    ids = []


    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        filename = os.path.basename(image_path)
        try:
            current_id = int(filename.split('-')[1])
        except (ValueError, IndexError):
            print(f"Peringatan: Melewatkan file dengan format nama yang salah: {filename}")
            continue

        face_samples.append(img)
        ids.append(current_id)

    print("[INFO] Data gambar dan label berhasil disiapkan.")
    return face_samples, np.array(ids)

recognizer = cv2.face.LBPHFaceRecognizer.create()

if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
    print(f"[ERROR] Folder '{dataset_path}' tidak ditemukan atau kosong.")
    print("Silakan jalankan script pembuat dataset terlebih dahulu.")
else:
    print("\n[INFO] Memulai training model. Proses ini mungkin memakan waktu beberapa saat...")
    
    faces, ids = get_images_and_labels(dataset_path)
    
    recognizer.train(faces, ids)
    
    recognizer.write(model_path)
    
    print(f"\n[SUCCESS] Training selesai. Model disimpan sebagai '{model_path}'")