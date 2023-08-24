import os
import time
import xml.etree.ElementTree as ET
import pandas as pd
import params
import tqdm
import numpy as np


## https://blog.paperspace.com/train-yolov5-custom-data/#convert-the-annotations-into-the-yolo-v5-format
def extract_data_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        data.append((name, xmin, ymin, xmax, ymax))

    return data


def convert_to_dataframe(data, xml_file):
    image_id = os.path.splitext(os.path.basename(xml_file))[0]
    rows = []
    for i, (name, xmin, ymin, xmax, ymax) in enumerate(data, start=1):
        rows.append((f"{image_id}", name, f"{xmin} {ymin} {xmax} {ymax}"))
    df = pd.DataFrame(rows, columns=['ImageId', 'ClassId', 'EncodedPixels'])

    return df


## xmin,ymin,xmax,ymax형태
def create_bbox_dataframe_minmax(df):
    # 새로운 DataFrame 생성
    new_df = pd.DataFrame(columns=['ImageId', 'width', 'height', 'bbox', 'source'])

    # 기존 DataFrame을 반복(iterate)하면서 처리
    for index, row in df.iterrows():
        image_id = row['ImageId']
        class_id = row['ClassId']
        encoded_pixels = row['EncodedPixels']
        bbox = [int(coord) for coord in encoded_pixels.split()]  # bbox 정보 파싱
        width = params.width  # 이미지의 너비
        height = params.height  # 이미지의 높이

        # 새로운 행(row) 추가
        new_row = {'ImageId': image_id, 'width': width, 'height': height, 'bbox': bbox, 'source': class_id}
        new_df = new_df.append(new_row, ignore_index=True)

    return new_df


## xcenter,ycenter,폭,높이 형태
def create_bbox_dataframe_yolo(df):
    # 새로운 DataFrame 생성
    new_df = pd.DataFrame(columns=['ImageId', 'width', 'height', 'x_center', 'y_center', 'classes'])

    for index, row in df.iterrows():
        image_id = row['ImageId']
        source = row['ClassId']
        encoded_pixels = row['EncodedPixels']
        bbox = [int(coord) for coord in encoded_pixels.split()]
        width = (bbox[2] - bbox[0]) / params.width  # 이미지의 너비
        height = (bbox[3] - bbox[1]) / params.height  # 이미지의 높이
        x_center = ((bbox[2] + bbox[0]) / 2) / params.width
        y_center = ((bbox[3] + bbox[1]) / 2) / params.height

        # 새로운 행(row) 추가
        new_row = {'image_id': image_id, 'width': width, 'height': height, 'x_center': x_center, 'y_center': y_center,
                   'classes': source}

        new_df = new_df.append(new_row, ignore_index=True)

        new_dict = {v: k for k, v in params.label_dic.items()}
        new_df['classes'] = new_df['classes'].replace(new_dict)

    return new_df


# xml_root = params.root_dir + '/Annotations'
# image_root = params.root_dir + '/Images'
# xml_list = os.listdir(xml_root)

xml_root = params.copy_data_path + '/labels/train'
image_root = params.copy_data_path + '/images/train'
xml_list = os.listdir(xml_root)

# xml_file = [os.path.join(xml_root, file) for file in xml_list] <- 잘되나 파일 하나만 뽑아서 테스트
# xml_file = os.path.join(xml_root, xml_list[0])
# data = extract_data_from_xml(xml_file)
# df = convert_to_dataframe(data, xml_file)

combined_df = pd.DataFrame()

for i in xml_list:
    xml_file = os.path.join(xml_root, i)
    data = extract_data_from_xml(xml_file)
    df = convert_to_dataframe(data, xml_file)
    combined_df = combined_df.append(df, ignore_index=True)

combined_df.to_csv('anno_230721_bbox.csv', index=False)
combined_df = pd.read_csv('anno_230721_bbox.csv')
print(combined_df)

# df = create_bbox_dataframe_minmax(df)
df = create_bbox_dataframe_yolo(combined_df)
print(df)

df.to_csv('anno_230721_yolo.csv', index=False)
