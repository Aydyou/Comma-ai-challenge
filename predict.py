import cv2
import numpy as np
import imutils
import tensorflow as tf

cap = cv2.VideoCapture('test.mp4')

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

np.save('testarray', buf)


diff_buf = [abs(buf[i][:][:][:]-buf[i+1][:][:][:]) for i in range(0,frameCount-1)]
diff_buf.append(diff_buf[frameCount-2][:][:][:])
diff_buf = np.array(diff_buf)

model=tf.keras.models.load_model('saved_model\my_model')

y=model.predict(diff_buf)

l=[]
for i in y:
    l.append(i[0])

with open('predict.txt', mode='wt', encoding='utf-8') as myfile:
    for lines in l:
        myfile.write((str(lines))+'\n')