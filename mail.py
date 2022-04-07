from sre_constants import SUCCESS
import cv2
from matplotlib.pyplot import fill_between
import mediapipe as mp

cap = cv2.VideoCapture(0)    #подключение камеры

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity = 0,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
    max_num_hands=2)
mpVisual = mp.solutions.drawing_utils
fing_Code = [(8 ,6), (12,10), (16,14), (20,18)]
thumb_coord = (4,2)

while cap.isOpened():
    succes, image = cap.read()
    if not succes:
        print("Не удалось подкглючить камеру")
        continue
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)
    upCount = 0

    if result.multi_hand_landmarks:
        
        miltiLandMarks =result.multi_hand_landmarks
        for idx, handLms, in enumerate(miltiLandMarks):
            inl = result.multi_handedness[idx].classification[0].label
            #print(inl)

        for handLms in miltiLandMarks:
            mpVisual.draw_landmarks(image, handLms,mp_hands.HAND_CONNECTIONS)  
            fingerlist = [] #список пальцев
            for idx, lm in enumerate(handLms.landmark):
                h, w , c = image.shape
                cx, cy = int(lm.x *w), int(lm.y * h)
                fingerlist.append((cx,cy))
            for coordinate in fing_Code:
                if fingerlist[coordinate[0]][1] < fingerlist[coordinate[1]][1]:
                    upCount += 1
            if fingerlist[thumb_coord[0]][0] < fingerlist[thumb_coord[1]][0]:
                upCount += 1
            
        cv2.putText(image, str(upCount), (50, 250), cv2.FONT_HERSHEY_COMPLEX, 12 , (0,200 ,50), 10)

        print(upCount)


    cv2.imshow('Video', image)

    key_press = cv2.waitKey(10) & 0xFF == 27

    if key_press == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()