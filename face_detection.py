
#pip install dlib

import time
import dlib
import cv2
from google.colab.patches import cv2_imshow #using gg colab


image = cv2.imread('/image.jpg')


hog_face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1('/content/mmod_human_face_detector.dat')

start = time.time()
faces_hog = hog_face_detector(image, 1)
end = time.time()
print("Hog + SVM Execution time: " + str(end-start))

for face in faces_hog:
  x = face.left()
  y = face.top()
  w = face.right() - x
  h = face.bottom() - y

  cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)


start = time.time()
faces_cnn = cnn_face_detector(image, 1)
end = time.time()
print("CNN Execution time: " + str(end-start))

for face in faces_cnn:
  x = face.rect.left()
  y = face.rect.top()
  w = face.rect.right() - x
  h = face.rect.bottom() - y

  cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)

cv2_imshow(image) # can change to cv2.imshow()
cv2.waitKey(0)