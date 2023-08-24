import os
import pandas as pd
import params
import preprocessing as pp
# import make_json as js
import subprocess
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.cuda.empty_cache()
torch.cuda.device(1)

df = pd.read_csv("anno_230721_yolo.csv")
index = list(set(df.image_id))

# pp.data_copy(index, df)
pp.yaml_dataset()

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cmd_detect_test = f'python {params.script_path_detect} --weights {params.weight_path} --img {params.img} --conf {params.conf} --iou-thres {params.iou} --source {params.test_data_path} --save-txt --save-conf'
subprocess.call(cmd_detect_test, shell=True)

# js()

