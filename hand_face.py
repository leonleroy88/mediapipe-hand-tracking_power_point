import cv2
import mediapipe as mp

# Modules
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Webcam non detectee")
    exit()

# Variables pour détecter le geste "coucou"
historique_x = []
coucou_cooldown = 0

def detecter_coucou(historique_x):
    """Détecte un mouvement droite -> gauche avec la main"""
    if len(historique_x) < 15:
        return False
    debut = sum(historique_x[:5]) / 5
    fin = sum(historique_x[-5:]) / 5
    deplacement = debut - fin  # positif = va vers la gauche
    return deplacement > 0.15  # seuil de mouvement

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- DETECTION MAIN ---
    results_hands = hands.process(rgb)

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, "MAIN DETECTEE", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Position X du poignet (point 0)
            x_poignet = hand_landmarks.landmark[0].x
            historique_x.append(x_poignet)
            if len(historique_x) > 30:
                historique_x.pop(0)

            # Détection coucou
            if coucou_cooldown == 0 and detecter_coucou(historique_x):
                print("👋 COUCOU détecté !")
                cv2.putText(frame, "COUCOU !", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                historique_x.clear()
                coucou_cooldown = 60  # évite les répétitions (60 frames)
    else:
        historique_x.clear()

    if coucou_cooldown > 0:
        coucou_cooldown -= 1

    # --- DETECTION VISAGE ---
    results_face = face.process(rgb)

    if results_face.detections:
        for detection in results_face.detections:
            mp_drawing.draw_detection(frame, detection)
        cv2.putText(frame, "VISAGE DETECTE", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

    cv2.imshow("Projet IA - Hand + Face Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    if cv2.getWindowProperty("Projet IA - Hand + Face Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()