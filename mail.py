from sre_constants import SUCCESS
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)    #подключение камеры

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands = 1,
    model_complexity = 0,
    min_tracking_confidence = 0.5,
    min_detection_confidence = 0.5)
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
    
    if multiLandMarks:
        for id, handLms, in enumerate(multiLandMarks):
            inl = result.multi_handedness[id].classification[0].label
            print(inl)
        upCount = 0
        for Handing in multiLandMarks: 
            mpVisual.draw_landmarks(image, Handing, mp_hands.Hand_CONNECTIONS)
            list = []
            for id, lane in enumerate(handLms.landmark):
                he, wi, l = image.shape
                x, y  = int(lane.x * wi), int(lane.y * he)
                list.append((x, y))
            for cord in fing_Code:
                if list[cord[0]][1] < list[cord[1]][1]:
                    upCount += 1
            if list[thumb_coord[0]][0] > list[thumb_coord[1]][0]:
                    upCount += 1

cap.release()

cv2.destroyAllWindows()