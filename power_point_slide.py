import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False

# Modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Webcam non detectee")
    exit()

# Variables
historique_x = []
cooldown = 0
dernier_geste = ""
derniere_action_temps = 0

def compter_doigts(hand_landmarks):
    """Compte le nombre de doigts levés"""
    tips = [8, 12, 16, 20]  # Index, Majeur, Annulaire, Auriculaire
    count = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    # Pouce
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        count += 1
    return count

def detecter_swipe(historique_x):
    """Retourne 'droite', 'gauche' ou None"""
    if len(historique_x) < 8:
        return None
    debut = sum(historique_x[:5]) / 5
    fin = sum(historique_x[-5:]) / 5
    deplacement = fin - debut

    if deplacement > 0.07:
        return "droite"
    elif deplacement < -0.07:
        return "gauche"
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # Affichage UI
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(frame, "CONTROLEUR POWERPOINT", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_poignet = hand_landmarks.landmark[0].x
            historique_x.append(x_poignet)
            if len(historique_x) > 30:
                historique_x.pop(0)

            doigts = compter_doigts(hand_landmarks)

            # --- DETECTION GESTE ---
            maintenant = time.time()

            if cooldown == 0:

                # Poing fermé = stop
                if doigts == 0:
                    dernier_geste = "✊ PAUSE"
                    cv2.putText(frame, "PAUSE", (20, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                # Main ouverte = rien
                elif doigts == 5:
                    dernier_geste = "✋ EN ATTENTE"

                # Swipe détecté
                else:
                    swipe = detecter_swipe(historique_x)
                    if swipe == "droite" and maintenant - derniere_action_temps > 1:
                        pyautogui.press('right')
                        dernier_geste = "➡️  SLIDE SUIVANTE"
                        print("➡️  Slide suivante")
                        historique_x.clear()
                        cooldown = 40
                        derniere_action_temps = maintenant

                    elif swipe == "gauche" and maintenant - derniere_action_temps > 1:
                        pyautogui.press('left')
                        dernier_geste = "⬅️  SLIDE PRECEDENTE"
                        print("⬅️  Slide précédente")
                        historique_x.clear()
                        cooldown = 40
                        derniere_action_temps = maintenant

    else:
        historique_x.clear()

    if cooldown > 0:
        cooldown -= 1

    # Afficher dernier geste
    if dernier_geste:
        cv2.putText(frame, dernier_geste, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Controleur PowerPoint - Gestes", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    if cv2.getWindowProperty("Controleur PowerPoint - Gestes", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()