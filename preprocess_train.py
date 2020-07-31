import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('train.mp4')

cap.set(3, 320)
cap.set(4,240)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


buf = np.empty((frameCount, frameHeight//2, frameWidth//2), np.dtype('uint8'))



fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=320)
    buf[fc] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fc += 1

cap.release()

buf=np.expand_dims(buf, axis=3)
buf=buf.astype('float16')
buf=buf/255.0


np.save('trainarray', buf)