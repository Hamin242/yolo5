import os

import pandas as pd

import json
import params

label_dic = params.label_dic
standard_label_path = r'D:/hm/Projoects/anno2anno/anno_sample_0710.csv'

# 라벨 딕셔너리 역방향 생성
reversed_label_dic = {value: key for key, value in label_dic.items()}
yolo_result_dir = r'yolov5/runs/detect/exp7/labels'

# 표준 레이블 데이터 로드 및 라벨 변환
standard_label_data = pd.read_csv(standard_label_path)

standard_label_data['classes'] = standard_label_data['classes'].replace(reversed_label_dic)

# 이미지 프리픽스 생성
standard_label_data['image_prefix'] = standard_label_data['image_id'].str[:2]

# YOLO 결과 파일 목록 가져오기
yolo_result_files = os.listdir(yolo_result_dir)

# 그룹화된 클래스 데이터 생성
grouped_classes = standard_label_data.groupby('image_id')['classes'].apply(list).reset_index()

# 이미지 프리픽스별 클래스 집합 생성
sample_grouped = standard_label_data.groupby('image_prefix')['classes'].apply(set).apply(sorted).reset_index()

# 빈 DataFrame을 생성하고 컬럼을 정의합니다.
tmp_df = pd.DataFrame(columns=['image_id', 'parts'])

# yolo_result_files 리스트에 있는 각 파일에 대해 반복합니다.
for i in yolo_result_files:
    # 파일 이름을 "_"로 분리하여 필요한 정보를 추출합니다.
    name_parts = i.split('_')
    image_id = name_parts[2].split('.')[0]
    model_id = name_parts[0]
    # 새로운 파일 이름을 생성합니다.
    new_file_name = f'{model_id}_{image_id}.bmp'

    yolo_result_path = os.path.join(yolo_result_dir, i)

    try:
        # YOLO 결과 파일을 열고 줄별로 처리합니다.
        with open(yolo_result_path, 'r') as file:
            lines = file.readlines()
            extracted_values = []
            for line in lines:
                # 각 줄을 공백으로 분리하여 값을 추출하고 첫 번째 값을 정수로 변환하여 리스트에 추가합니다.
                values = line.strip().split()
                # 추출한 값을 새로운 행으로 DataFrame에 추가합니다.
                # ignore_index=True로 설정하여 인덱스를 자동으로 조정합니다.
                tmp_df = tmp_df.append({'image_id': new_file_name, 'parts': int(values[0])}, ignore_index=True)
    except FileNotFoundError as e:
        # 파일이 찾을 수 없는 경우, 예외를 처리하고 넘어갑니다.
        pass

grouped = tmp_df.groupby('image_id')['parts'].apply(set).apply(sorted).reset_index()

# JSON 형식으로 저장할 데이터를 담을 리스트 생성
unchecked_data = []

for i in range(len(grouped)):
    try:
        tmp = sample_grouped.loc[sample_grouped['image_prefix'] == grouped['image_id'][i][:2], 'classes'].values
        image_id = grouped['image_id'][i].replace("_", "").replace(".bmp", "")

        if len(set(tmp[0]) - set(grouped['parts'][i])) == 0:
            print(image_id, 'is OK')
        else:
            # 미검 데이터 정보를 딕셔너리로 생성
            unchecked_item = {
                "root_number": image_id,
                "reasons": []
            }
            for j in set(tmp[0]) - set(grouped['parts'][i]):
                unchecked_item["reasons"].append({
                    "part": label_dic[j],
                    "status": "미검"
                })
            # JSON 데이터에 추가
            unchecked_data.append(unchecked_item)
    except IndexError as e:
        print("그룹화된 데이터에 대한 샘플 데이터 없음:", grouped['image_id'][i][:2])
        pass

# JSON 파일로 저장 (한국어 인코딩 적용)
with open('unchecked_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(unchecked_data, json_file, ensure_ascii=False, indent=4)

print("미검 데이터를 JSON 파일로 저장했습니다.")