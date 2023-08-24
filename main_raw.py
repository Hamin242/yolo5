##https://www.kaggle.com/code/palitaboonkuea/yolov5-globalwheat

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
import gc
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.cuda.empty_cache()
torch.cuda.device(1)

df = pd.read_csv(r"C:\Users\hm\Desktop\tmp\Mask_rCNN\anno_bbox_fi.csv")

bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

for i, column in enumerate(['x', 'y', 'width', 'height']):
    df[column] = bboxs[:, i]
df.drop(columns=['bbox'], inplace=True)
df['x_center'] = df['x'] + df['width'] / 2
df['y_center'] = df['y'] + df['height'] / 2
df['classes'] = df['source']
df['x_center'] = df['x_center'] / 2464
df['y_center'] = df['y_center'] / 2056
df['width'] = df['width'] / 2464
df['height'] = df['height'] / 2056
df = df[['image_id', 'width', 'height', 'x_center', 'y_center', 'classes']]

new_dict = {v: k for k, v in params.label_dic.items()}
df['classes'] = df['classes'].replace(new_dict)

index = list(set(df.image_id))

if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    print("Number OF Trainning Images:" + str(len(os.listdir(params.train_raw_image_path))))
    print("Number OF Testing Images:" + str(len(os.listdir(params.test_raw_image_path))))

    source = 'train'

    # if True:
    #     for fold in [0]:
    #         val_index = index[len(index) * fold // 5:len(index) * (fold + 1) // 5]
    #         for name, mini in tqdm(df.groupby('image_id')):
    #             if name in val_index:
    #                 path2save = '/val/'
    #             else:
    #                 path2save = '/train/'
    #             if not os.path.exists('C:/Users/hm/Desktop/tmp/Data/labels/' + path2save):
    #                 os.makedirs('C:/Users/hm/Desktop/tmp/Data/labels/' + path2save)
    #             try:
    #                 with open('C:/Users/hm/Desktop/tmp/Data/labels/' + path2save + name + ".txt", 'w+') as f:
    #                     row = mini[['classes', 'x_center', 'y_center', 'width', 'height']].astype(float).values
    #                     row = row.astype(str)
    #                     for j in range(len(row)):
    #                         text = ' '.join(row[j])
    #                         f.write(text)
    #                         f.write(" \n  ")
    #             except FileNotFoundError:
    #                 pass
    #
    #             if not os.path.exists('C:/Users/hm/Desktop/tmp/Data/images/{}'.format(path2save)):
    #                 os.makedirs('C:/Users/hm/Desktop/tmp/Data/images/{}'.format(path2save))
    #
    #             try:
    #                 sh.copy("D:/hm/Projoects/SampleData_230607/train/{}.bmp".format(name),
    #                         'C:/Users/hm/Desktop/tmp/Data/images/{}/{}.bmp'.format(path2save, name))
    #             except FileNotFoundError:
    #                 pass

    Data_Path = "hm/Projoects/Mask-RCNN/Data"

    with open(f"D:/hm/Projoects/Mask-RCNN/yolov5/EEI_yolov5.yaml", "w+") as file_:
        file_.write(
            f"""
               train: /{Data_Path}/images/train  
               val: /{Data_Path}/images/val  
               test:  # test images (optional)
               nc: 102
               names: {params.label_dic}
                """
        )

    # cmd_train = f'python {params.script_path_train} --batch {params.batch} --epochs {params.epochs} --device 1 --data {params.yaml_path} --cfg {params.cfg_path} --name {params.name_model} --patience 300'
    # subprocess.call(cmd_train, shell=True)

    cmd_detect = f'python {params.script_path_detect} --weights {params.weight_path} --img {params.img} --conf {params.conf} --iou-thres {params.iou} --source {params.test_raw_image_path} --save-txt --save-conf'
    subprocess.call(cmd_detect, shell=True)

    # predicted_files = []
    # for (dirpath, dirnames, filenames) in os.walk("/kaggle/working/yolov5/runs/detect/exp"):
    #     predicted_files.extend(filenames)

    # !ls -R runs/detect/exp

