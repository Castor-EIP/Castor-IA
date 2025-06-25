from ultralytics import YOLO
import cv2
import numpy as np

# Charger le modèle pré-entraîné (YOLOv11 nano pour de bonnes perfs en temps réel)
model = YOLO("yolo11n.pt")

cap1 = cv2.VideoCapture("http://10.67.164.84:4747/video")
cap2 = cv2.VideoCapture(1)

# Obtenir l'index de la classe 'cell phone'
cell_phone_index = list(model.names.values()).index('cell phone')

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # Activer le mode miroir pour les frames
    frame1 = cv2.flip(frame1, 1)
    frame2 = cv2.flip(frame2, 1)

    results1 = model(frame1)  # Détection des objets pour la première caméra
    results2 = model(frame2)  # Détection des objets pour la deuxième caméra

    annotated_frame1 = results1[0].plot()  # Ajouter les boxes détectées pour la première caméra
    annotated_frame2 = results2[0].plot()  # Ajouter les boxes détectées pour la deuxième caméra

    # Vérifier la taille des frames annotées avant de les combiner
    height1, width1 = annotated_frame1.shape[:2]
    height2, width2 = annotated_frame2.shape[:2]

    # Redimensionner les frames annotées pour qu'elles aient la même hauteur
    if height1 > height2:
        scale = height1 / height2
        annotated_frame2 = cv2.resize(annotated_frame2, (int(width2 * scale), height1))
    elif height2 > height1:
        scale = height2 / height1
        annotated_frame1 = cv2.resize(annotated_frame1, (int(width1 * scale), height2))

    # Combiner les deux frames annotées côte à côte
    combined_frame = np.hstack((annotated_frame1, annotated_frame2))

    # Détection des téléphones portables
    cell_phones1 = sum(1 for obj in results1[0].boxes.data if int(obj[-1]) == cell_phone_index)
    cell_phones2 = sum(1 for obj in results2[0].boxes.data if int(obj[-1]) == cell_phone_index)

    # Afficher la caméra avec le plus de téléphones portables détectés
    if cell_phones1 > cell_phones2:
        prioritized_frame = annotated_frame1
    else:
        prioritized_frame = annotated_frame2

    # Redimensionner la frame priorisée pour correspondre à la largeur de la frame combinée
    prioritized_frame = cv2.resize(prioritized_frame, (combined_frame.shape[1], prioritized_frame.shape[0]))

    # Combiner les frames annotées et la frame priorisée
    final_frame = np.vstack((combined_frame, prioritized_frame))

    cv2.imshow("YOLO Detection", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()