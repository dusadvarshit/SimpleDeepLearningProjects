import streamlit as st
import os
import numpy as np
import cv2
import tensorflow as tf 
from pages.submodules.cartoonize import network, guided_filter
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray



def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image
    

def cartoonize(image, model_path):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])

    network_out = network.unet_generator(input_photo)
        
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    
    # try:
    image = resize_crop(image)
    batch_image = image.astype(np.float32)/127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)
    output = sess.run(final_out, feed_dict={input_photo: batch_image})
    output = (np.squeeze(output)+1)*127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output
    # except:
    #     print('cartoonize failed')


st.title('Cartoonize Your Image')
st.subheader('Only jpeg, jpg and png allowed')

img_file = st.file_uploader(label="Upload your picture", type=['jpg', 'jpeg', 'png'])

model_path = "pages/submodules/cartoonize/saved_models"

flag = False

if img_file!= None:
    st.subheader("Your Image")
    st.image(img_file)

    image = Image.open(img_file)
    img_array = np.array(image)

    cartoon_subheader = st.subheader("Wait for Cartoonized Image")

    cartoon_file = cartoonize(img_array, model_path)
    flag = True


if flag== True:
    cartoon_subheader.subheader('Here is your Cartoonized Image')
    st.image(cartoon_file)








