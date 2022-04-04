import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)    #подключение камеры

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 1)
mpVisual = mp.solutions.draw