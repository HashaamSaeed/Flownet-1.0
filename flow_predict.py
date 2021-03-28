
import os
import random
import numpy as np
from tqdm import tqdm 

import sys
import glob
import cv2
import tensorflow as tf
import scipy.misc
import matplotlib
import time
import re
import argparse

import keras
from keras.models import Model
from keras import backend as K

from Ops_Utils_flowlib_show import show_flow , flow_to_image

from PIL import Image


from flowiz_new import convert_from_flow
#from EPE_loss import EPE
#import tensorflow as tf


model_prediction = keras.models.load_model('FLowNet_S_1',compile=False)

## example of input path 
## path_image_1/path_image_2 = "image.png"

def visualise_predicted_image(path_image_1 , path_image_2):
	

	img1 = cv2.imread(path_image_1)
	img2 = cv2.imread(path_image_2)

	img_cat = np.concatenate((img1,img2),axis=2)
	img_cat =  np.expand_dims(img_cat,axis =-4)

	model_pred = model_prediction.predict(img_cat,verbose=1)

	#print(model_pred.shape)

	model_pred = np.squeeze(model_pred)
	#print(model_pred.shape)
	show_flow(model_pred)



## example of input path 
## input_path = "Jets_video.avi"
## predicted video is saved in the same folder as this .py file 

def visualise_predicted_video(input_video_path):

	output_path = "flow_visualise_video.avi"
	frames_vid = []

	cap = cv2.VideoCapture(input_video_path)

	fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
	out = cv2.VideoWriter(output_path  , fourcc, 30, (512,384))

	while cap.isOpened():

	    
	    ret, frame = cap.read()
	    # if frame is read correctly ret is True
	    
	    if not ret:
	        print("Can't receive frame (stream end?). Exiting ...")

	        break
	    
	    frames_vid.append(frame)



	frames_vid = np.asarray(frames_vid)
	if len(frames_vid) % 2 == 0 :
		frames_vid = frames_vid
	else:
		frames_vid = frames_vid[0:len(frames_vid)-1,:,:,:]

	vid_load = []

	for i in tqdm(range(0,len(frames_vid),2)):
		t0_frame =frames_vid[i,:,:,:]
		t1_frame =frames_vid[i+1,:,:,:]
		vid_cat = np.concatenate((t0_frame,t1_frame),axis=2)
		vid_load.append(vid_cat)

		#if i <= len(frames_vid) - 2:

	vid_load = np.asarray(vid_load)	
	print('vid_load_shape',vid_load.shape)

	flo_vid	= []

	#show_vid = cv2.imshow('hey', np.squeeze(vid_load[1,:,:,3:7]))
	#cv2.waitKey(0)
	#print('len vid_load',len(vid_load))
			
	for i in tqdm(range(len(vid_load))):
		
		imagine = vid_load[i,:,:,:]
		imagine = np.expand_dims(imagine,axis=-4)
		#print('imagine',imagine.shape)
		vid_to_flow_pred = model_prediction.predict(imagine,verbose=0)
		#helloworld = convert_from_flow(np.squeeze(vid_to_flow_pred))
		helloworld = flow_to_image(np.squeeze(vid_to_flow_pred))
		flo_vid.append(helloworld)



	flo_vid = np.asarray(flo_vid)
	print("flo_vid shape",flo_vid.shape)

	#newowowo = cv2.imshow('helll',np.squeeze(flo_vid[1,:,:,:]))
	#cv2.waitKey(0)
	#print("convert_show_flow",convert_show_flow.shape)

	for i in tqdm(range(len(flo_vid))):
		new_flo = np.squeeze(flo_vid[i,:,:,:])
		#new_flo_show = cv2.imshow('show',new_flo)
		#cv2.waitKey(0)
		#print('new flo ',new_flo.shape)
		out.write(new_flo)


	cap.release()
	out.release()
	cv2.destroyAllWindows()




#visualise_predicted_video("cars moving.avi")
visualise_predicted_image("000171.png","00017.png")