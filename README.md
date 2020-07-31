# Comma-ai-challenge
This is an attempt to tackle the comma ai speed detection [challenge](https://github.com/commaai/speedchallenge). You can find the training and test videos in their link. 

By running `preprocess_train.py` the training video is turned into 240p grayscale numpy array and saved in the folder. In `train.py` we split the training video into two sections. First 80%
is reserved for training and the remaining 20% for validation purpose. The input to the model is the difference of two consecutive frames and the model itself is a layer of CNNs.
The callback function is written to drop training once validation MSE gets below 4. In this model the validation MSE is about 3.4. You can load the model with a command like `model=tf.keras.models.load_model('saved_model\my_model')`.
In `predict.py` we turn the testing video to the same format as training and give the predictions and save them as a text file. You can find the results in `predict.txt`. The `train_speed_prediction.txt`
contains the speed predictions for the training video, you can compare it with `speed.txt` which contains the original speeds. Using this model the lowest validation MSE that I was able to achieve
was about 3.4. It seems in order to get better results more work is required. An idea that I might try in the future is doing image segmentation on the video and separating the road from rest
of the picture and performing the model on road only section. Another thing that could be done is applying some sort of smoothing, like averaging the speed to remove the jitteriness of the speed predictions.

In order to run the code the following are required:
`opencv, imutils, tensorflow`
