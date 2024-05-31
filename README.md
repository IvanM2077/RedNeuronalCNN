# Red neuronal convolucional CNN
## Proyecto escolar del entrenamiento de una red neuronal con una base de datos propia.
El proyecto en cuestión tiene como objetivo el entrenamiento de una red neuronal
para el reconocimiento de imagenes a partir de una base de datos con fines eductativos 
sobre la importancia que tiene el reconocimiento de patrones a partir de diversos, métodos. Siendo
la red neuronal la mejor para reconocer patrones sin perder espacialidad en comparación a los metodos
de SVM y DecisionTree. 
### Instalacion de las librerías para el entrenamiento de la red
- from google.colab import drive
- import os
- import glob
- import numpy as np
- import matplotlib.pyplot as plt
- import pandas as pd
- import cv2 as cv
- import random
- from numpy import asarray, save, load, random
- from keras.datasets import mnist
- from keras.models import Sequential, load_model
- from keras.layers import Dense, Dropout, Flatten
- from tensorflow.keras.layers import Conv2D, MaxPool2D
- from tensorflow.keras.utils import to_categorical, array_to_img, img_to_array, load_img
- from sklearn.model_selection import train_test_split
- import h5py

### Uso
El uso de este apliación es para el entrenamiento de un conjunto de imagenes 
que hacen uso de la librería de mediapipe para la esqueletización de la mano
y ayudar al modelo aprender mejor el reconocimiento de patrones. 



