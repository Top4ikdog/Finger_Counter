from sre_constants import SUCCESS
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)    #подключение камеры

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 2)
mpVisual = mp.solutions.drawing_utils
fing_Code = [(8 ,6), (12,10), (16,14), (20,18)]
thumb_coord = (4,2)

while cap.isOpened():
    succes, image = cap.read()
    if not succes:
        print("Не удалось подкглючить камеру")
        continue
    
    image = cv2.flip(image, 1)
    cv2.imshow('Video', image)

    key_press = cv2.waitKey(10)

    if key_press == ord('q'):
        break

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)
    multiLandMarks = result.multi_hand_landmarks
    
    upCount = 0 # счет пальцев

    if multiLandMarks:
        for idx, handLms, in enumerate(multiLandMarks):
            inl = result.multi_handedness[idx].classification[0].label
            print(inl)
        
        for handLms in multiLandMarks:
            mpVisual.draw_landmarks(image, handLms,mp_hands.HAND_CONNECTIONS)  
            list = [] #список пальцев
            for idx, lm in enumerate(handLms.landmark):
                h, w , c = image.shape
                cx, cy = int(lm.x *w), int(lm.y * h)
                list.append((cx,cy))
            for coordinate in fing_Code:
                if list[coordinate[0]][1] < list[coordinate[1][1]]:
                    upCount =+ 1


cap.release()

cv2.destroyAllWindows()