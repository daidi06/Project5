# -*- coding: utf-8 -*-

import io
import tkinter as tk
from tkinter import filedialog
import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import inception_resnet_v2

from google_drive_downloader import GoogleDriveDownloader as gdd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model_id = '1l2ohQLtlPXgbWH6JGzPAMxMTR5o9Mpcx'


lab_dic = {'Afghan_hound': 0, 'African_hunting_dog': 1,  'Airedale': 2, 'American_Staffordshire_terrier': 3,
 'Appenzeller': 4, 'Australian_terrier': 5, 'Bedlington_terrier': 6, 'Bernese_mountain_dog': 7,
 'Blenheim_spaniel': 8, 'Border_collie': 9, 'Border_terrier': 10, 'Boston_bull': 11, 'Bouvier_des_Flandres': 12, 'Brabancon_griffon': 13, 'Brittany_spaniel': 14, 'Cardigan': 15, 'Chesapeake_Bay_retriever': 16, 'Chihuahua': 17, 'Dandie_Dinmont': 18, 'Doberman': 19, 'English_foxhound': 20, 'English_setter': 21, 'English_springer': 22, 'EntleBucher': 23, 'Eskimo_dog': 24, 'French_bulldog': 25, 'German_shepherd': 26, 'German_short-haired_pointer': 27, 'Gordon_setter': 28, 'Great_Dane': 29, 'Great_Pyrenees': 30, 'Greater_Swiss_Mountain_dog': 31, 'Ibizan_hound': 32, 'Irish_setter': 33, 'Irish_terrier': 34, 'Irish_water_spaniel': 35, 'Irish_wolfhound': 36, 'Italian_greyhound': 37, 'Japanese_spaniel': 38, 'Kerry_blue_terrier': 39, 'Labrador_retriever': 40, 'Lakeland_terrier': 41, 'Leonberg': 42,
 'Lhasa': 43, 'Maltese_dog': 44, 'Mexican_hairless': 45, 'Newfoundland': 46, 'Norfolk_terrier': 47, 'Norwegian_elkhound': 48, 'Norwich_terrier': 49, 'Old_English_sheepdog': 50, 'Pekinese': 51, 'Pembroke': 52, 'Pomeranian': 53, 'Rhodesian_ridgeback': 54, 'Rottweiler': 55, 'Saint_Bernard': 56, 'Saluki': 57, 'Samoyed': 58, 'Scotch_terrier': 59, 'Scottish_deerhound': 60, 'Sealyham_terrier': 61, 'Shetland_sheepdog': 62, 'Shih-Tzu': 63, 'Siberian_husky': 64, 'Staffordshire_bullterrier': 65, 'Sussex_spaniel': 66, 'Tibetan_mastiff': 67, 'Tibetan_terrier': 68, 'Walker_hound': 69, 'Weimaraner': 70, 'Welsh_springer_spaniel': 71, 'West_Highland_white_terrier': 72, 'Yorkshire_terrier': 73, 'affenpinscher': 74, 'basenji': 75, 'basset': 76, 'beagle': 77, 'black-and-tan_coonhound': 78, 'bloodhound': 79, 'bluetick': 80, 'borzoi': 81, 'boxer': 82, 'briard': 83, 'bull_mastiff': 84, 'cairn': 85, 'chow': 86, 'clumber': 87, 'cocker_spaniel': 88, 'collie': 89, 'curly-coated_retriever': 90, 'dhole': 91, 'dingo': 92, 'flat-coated_retriever': 93, 'giant_schnauzer': 94, 'golden_retriever': 95, 'groenendael': 96, 'keeshond': 97, 'kelpie': 98, 'komondor': 99, 'kuvasz': 100, 'malamute': 101, 'malinois': 102, 'miniature_pinscher': 103, 'miniature_poodle': 104, 'miniature_schnauzer': 105, 'otterhound': 106, 'papillon': 107, 'pug': 108, 'redbone': 109, 'schipperke': 110, 'silky_terrier': 111, 'soft-coated_wheaten_terrier': 112, 'standard_poodle': 113, 'standard_schnauzer': 114, 'toy_poodle': 115, 'toy_terrier': 116, 'vizsla': 117, 'whippet': 118, 'wire-haired_fox_terrier': 119}


def load_image():
	print("Selectionner l'image a predire: ")
	filename = filedialog.askopenfilename()
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
	# prepare pixel data
	img = inception_resnet_v2.preprocess_input(img)
	return img

# load an image and predict the class
def run_example():
	# load the image
	img = load_image()
	# load model
	print("Selectionner le dossier du model")
	folder_selected = filedialog.askdirectory()
	gdd.download_file_from_google_drive(file_id = model_id, dest_path = folder_selected + '/model.h5', unzip=False)
	model = tf.keras.models.load_model(folder_selected + '/model.h5')
	# predict the class
	pred = model.predict(img)
	return pred
 
# entry point, run the example
pred = run_example()
val_pred = np.argmax(pred, axis=1)
race = list(lab_dic.keys())[list(lab_dic.values()).index(val_pred)]
proba = pred[0][val_pred] *100
print('La race du chien est: %s avec une probabilité de  (%.2f%%)' % (race, proba))

