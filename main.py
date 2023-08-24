##https://www.kaggle.com/code/palitaboonkuea/yolov5-globalwheat
import os
import pandas as pd

import params
import preprocessing as pp
import subprocess

import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.cuda.empty_cache()
torch.cuda.device(1)

## xml2yaml 따로 돌려야댐 // eei-update 폴더에서 데이터 이돟으로 xml 옮기고 진행함
## xml2txt로 기존의 xml 형태의 anno 값을 txt로 바꿔서 label에 넣어줘야댐
df = pd.read_csv("anno_230721_yolo.csv")
# df = pd.read_csv("anno_230630.csv")

index = list(set(df.image_id))

if __name__ == '__main__':
    # pp.data_copy(index, df)
    pp.yaml_dataset()
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    print("Number OF Train Images:" + str(len(os.listdir(params.copy_data_path+'/images/train'))))
    print("Number OF Val Images:" + str(len(os.listdir(params.copy_data_path+'/images/val'))))
    print("Number OF Test Images:" + str(len(os.listdir(params.copy_data_path+'/images/test'))))

    source = 'train'
    cmd_train = f'python {params.script_path_train} --batch {params.batch} --epochs {params.epochs} --device 1 --data {params.yaml_path} --cfg {params.cfg_path} --name {params.name_model} --patience 50 --img {params.img} --rect'
    subprocess.call(cmd_train, shell=True)

    cmd_detect_val = f'python {params.script_path_detect} --weights {params.weight_path} --img {params.img} --conf {params.conf} --iou-thres {params.iou} --source {params.val_data_path} --save-txt --save-conf'
    subprocess.call(cmd_detect_val, shell=True)

    #x테스트 할 때 이거 뺴고 다 꺼놔도댐
    cmd_detect_test = f'python {params.script_path_detect} --weights {params.weight_path} --img {params.img} --conf {params.conf} --iou-thres {params.iou} --source {params.test_data_path} --save-txt --save-conf'
    subprocess.call(cmd_detect_test, shell=True)

    ## 몇개 샘플링이 필요하면 원본 코드를 참고하자
    ### 5개의 CAM 부품 정보를 set에 쌓고 없는 데이터만 뽑고 해당 데이터를 출력
    #### 결과를 json 형태로 MFC어 던지긔

