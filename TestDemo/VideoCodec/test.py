import cv2

cap = cv2.VideoCapture('/home/chuhw/CV/DVC/TestDemo/VideoCodec/testVideo.mp4')
(flag,frame) = cap.read()
crop_size = (832, 448)
for i in range(0,31):
    
frame = cv2.resize(frame, crop_size, interpolation = cv2.INTER_CUBIC)
cv2.imwrite('./1.png',frame)