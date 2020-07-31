import numpy as np
import tensorflow as tf

buf=np.load('trainarray.npy')

diff_buf= [abs(buf[i][:][:][:]-buf[i+1][:][:][:]) for i in range(0,20399)]
diff_buf.append(diff_buf[20398][:][:][:])



buf_train=np.array(diff_buf[0:16320][:][:][:])
buf_validation=np.array(diff_buf[16320:20400][:][:][:])


labelstext=open('speed.txt', 'r')
labels=[]
for i in labelstext:
    labels.append(float(i))
labels=np.array(labels)

labels_train=labels[0:16320]
labels_validation=labels[16320:20400]

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_mse')<4.0):
      print("\nReached mse lower than 4")
      self.model.stop_training = True

callbacks = myCallback()

model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), input_shape=(240,320,1), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),


    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    #The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    #fifth
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    #sixth
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    #7th
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    # Flatten the results to feed into a DNN
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1, activation='linear')
])



model.compile(loss="mse",
              optimizer='adam',
              metrics=['mse','mae'])

model.fit(
      buf_train,
      labels_train,
      epochs=20,
      batch_size=20,
      validation_data=(buf_validation,labels_validation), verbose=1, callbacks=[callbacks])

model.save('saved_model\my_model')

y=model.predict(np.array(diff_buf))

l=[]
for i in y:
    l.append(i[0])

with open('train_speed_prediction.txt', mode='wt', encoding='utf-8') as myfile:
    for lines in l:
        myfile.write((str(lines))+'\n')