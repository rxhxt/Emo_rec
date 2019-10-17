import cv2
from keras.models import load_model
import time
import numpy as np


def pandas_vector_to_list(pandas_df):
    py_list = [item[0] for item in pandas_df.values.tolist()]
    return py_list

def get_op(op_list):
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    i = np.argmax(op_list)
    print(emotion_labels[i])
    return emotion_labels[i]


def process_pixels(pixels, img_size=48):
    print("hii2")
    pixels_as_list = pandas_vector_to_list(pixels)
    np_image_array = []
    for index, item in enumerate(pixels_as_list):
        data = np.zeros((img_size, img_size), dtype=np.uint8)
        for i in range(0, img_size):
            pixel_index = i * img_size
            data[i] = pixels_as_list[pixel_index:pixel_index + img_size]
        np_image_array.append(np.array(data))
    np_image_array = np.array(np_image_array)
    np_image_array = np_image_array.astype('float32') / 255.0
    return np_image_array

def pred(frame):
    frame = frame[np.newaxis,:,:,np.newaxis]
    temp = mod.predict(frame)
    print(temp)
    op = get_op(temp)
    return op



video = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
color = (0,255,0)
stroke = 4
filepath = "final_weights.h5"

mod = load_model(filepath, custom_objects=None, compile=True)
mod.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])


video.open(0)

print(video.isOpened())
while True:
    (check, frame) = video.read()
    # print(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,12)
    #    print(np.asarray(faces).shape)


    for(x_start,y_start,x_end,y_end) in faces:
        roi_gray = gray[y_start:y_start+y_end,x_start:x_start+x_end]
        roi_col = frame[y_start:y_start+y_end,x_start:x_start+x_end]
        resized = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)
        cv2.imshow("win22", resized)
        text = pred(np.array(resized)/255)
        frame2 = cv2.rectangle(frame,(x_start,y_start),(x_start+x_end,y_start+y_end),color,stroke)
        cv2.putText(img=frame2, text=text , fontFace=cv2.FONT_HERSHEY_COMPLEX, org=(x_start,y_start-10), fontScale=1, color=(0, 0, 0),
                    thickness=3)
    cv2.imshow('capturing', frame)
    Key = cv2.waitKey(1)
    if Key ==ord('q'):
        break

frm = np.array(frame)
print(frm.tolist() )

video.release()
cv2.destroyAllWindows()
