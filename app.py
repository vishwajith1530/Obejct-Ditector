import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly
from PIL import Image
st.title("Object Detector")

img_file = st.file_uploader("Upload an image")
st.write(type(img_file))


if img_file is not None:
    file_bytes = np.array(bytearray(img_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes,1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    st.image(opencv_image, channels="RGB")


generate_pred = st.button("Predict Object")
if generate_pred:
    bbox, label, conf = cv.detect_common_objects(img_file)
    output_image = draw_bbox(img_file,bbox,label,conf)
    #plt.imshow(output_image)
    st.image(output_image,channels="RGB")