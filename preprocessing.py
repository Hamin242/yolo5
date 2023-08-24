import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from tqdm import tqdm
import shutil as sh
import os
import pandas as pd
import tensorflow as tf
import keras as k
import params
import subprocess
from tensorflow.keras.utils import img_to_array, load_img


def data_copy(index,df):
    if True:
        for fold in [0]:
            val_index = index[len(index) * fold // 5:len(index) * (fold + 1) // 5]
            for name, mini in tqdm(df.groupby('image_id')):
                if name in val_index:
                    path2save = '/val/'
                else:
                    path2save = '/train/'
                if not os.path.exists(params.copy_data_path+'/labels/'+ path2save):
                    os.makedirs(params.copy_data_path+'/labels/'+ path2save)
                try:
                    with open(params.copy_data_path+'/labels/' + path2save + name + ".txt", 'w+') as f:
                        row = mini[['classes', 'x_center', 'y_center', 'width', 'height']].astype(float).values
                        row = row.astype(str)
                        for j in range(len(row)):
                            text = ' '.join(row[j])
                            f.write(text)
                            f.write(" \n  ")
                except FileNotFoundError:
                    pass

                if not os.path.exists(params.copy_data_path+'/images/{}'.format(path2save)):
                    os.makedirs(params.copy_data_path+'/images/{}'.format(path2save))

                try:
                    sh.copy(params.train_raw_image_path+"/{}.bmp".format(name),
                            params.copy_data_path+'/images/{}/{}.bmp'.format(path2save, name))
                except FileNotFoundError:
                    pass

def yaml_dataset():
    with open(params.yaml_path, "w+") as file_:
        file_.write(f"""                   
                   train: {params   .copy_data_path}/images/train 
                   val: {params.copy_data_path}/images/val  
                   test: {params.copy_data_path}/images/test
                   nc: {len(params.label_dic)}
                   names: {params.label_dic}
                    """
        )