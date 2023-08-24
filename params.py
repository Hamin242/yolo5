import os
import time
label_dic = {0: 'LockerCover_40273904', 1: 'Turbocharger_40376582', 2: 'Spacer_40309382', 3: 'SensorWaterTemp_40273637',
             4: 'ExhaustManifold_40268003', 5: 'ACVHoseB_40265762', 6: 'Spacer_40322591', 7: 'Flywheel_40370807',
             8: 'EGRCooler_40380976', 9: 'RearCase_40324088', 10: 'OilFiller_40265587', 11: 'CrankPulley_40265565',
             12: 'TimingCase_40273907', 13: 'WaterPump_40344260', 14: 'Heatsink_40288672', 15: 'OilFilter_40409065',
             16: 'OilCooler_40336395', 17: 'OilPan_40414815', 18: 'InjectionPump_40265601', 19: 'LockerCover_40306424',
             20: 'OilFilter_40318590', 21: 'InjectionPipe_40248301~3', 22: 'IntakeManifold_40306419',
             23: 'WaterPump_40376775', 24: 'OilFillerCover_40007385', 25: 'Flywheel_40389980', 26: 'Heatplug_40007624',
             27: 'RearCase_40277239', 28: 'CrankPulley_40233127', 29: 'TimingCase_40277238', 30: 'OilFiller_40395153',
             31: 'BearingCase_40009045', 32: 'OilPan_20021510', 33: 'ReliefValve_40006985', 34: 'Spacer_40309381',
             35: 'EGRValve_40446187', 36: 'ACVPipe_40446923', 37: 'ACV_40265758', 38: 'ACVHoseA_40322597',
             39: 'FuelFilter_40354756', 40: 'Flywheel_40446749', 41: 'EGRPassage_40446684', 42: 'RearCase_40446748',
             43: 'OilFiller_40332649', 44: 'OilPan_20165463', 45: 'SensorP2_40273639', 46: 'ExhaustManifold_40354759',
             47: 'Flywheel_40272348', 48: 'RearCase_40268001', 49: 'FuelFilter_40407354', 50: 'OilPan_40414814',
             51: 'Flywheel_40272344', 52: 'EGRCooler_40266106', 53: 'ExhaustManifold_40268002',
             54: 'LockerCover_40273903', 55: 'RearCase_40267499', 56: 'OilFiller_40414522', 57: 'CrankPulley_40009209',
             58: 'OilPan_40272349', 59: 'PreFilter_40266160', 60: 'Spacer_40439261', 61: 'RearCase_40312247',
             62: 'OilPan_40415242', 63: 'ACVPipe_40322598', 64: 'Flywheel_40322582', 65: 'RearCase_40322595',
             66: 'OilPan_20153073', 67: 'MainFilter_40407355', 68: 'Spacer_40294842', 69: 'InjectionPump_11111111',
             70: 'Flywheel_40272347', 71: 'RearCase_40267500', 72: 'WaterPump_40376807', 73: 'Spacer_40009955',
             74: 'IntakeManifold_40306416', 75: 'LockerCover_40306422', 76: 'InjectionPipe_40011025~8',
             77: 'Flywheel_40389977', 78: 'RearCase_20020362', 79: 'OilFiller_40326392', 80: 'TimingCase_40325870',
             81: 'OilPan_40315597', 82: 'Turbocharger_40436004', 83: 'ExhaustManifold_40008974',
             84: 'OilCooler_40007587', 85: 'RearCase_40279774', 86: 'IntakeManifold_40237804',
             87: 'LockerCover_40295008', 88: 'InjectionPipe_40215204~7', 89: 'ExhaustManifold_40220568',
             90: 'AirHeater_40217404', 91: 'Flywheel_40355151', 92: 'AirHeater_40376259', 93: 'OilFiller_40415846',
             94: 'RearCase_40298672', 95: 'OilPan_20022496', 96: 'Flywheel_40389981', 97: 'Flywheel_40353462',
             98: 'RearCase_40352881', 99: 'OilPan_20153074', 100: 'FuelFilter_40271220', 101: 'bg'}

root_dir = 'D:/Data/EEI/Training_data'

# 원본 이미지
width = 2464
height = 2056

train_raw_image_path = root_dir + '/Images'
test_raw_image_path = 'D:/Data/0_EEI_Test/TestSample_OK'
## Test가 다 쪼개져잇어서 합쳐줘서 폴더에 저장하기
device = 1

copy_data_path = 'D:/hm/Projoects/yolo5/data'

script_path_train = './yolov5/train.py'
yaml_path = 'EEI_yolov5.yaml'
cfg_path = './yolov5/models/yolov5s.yaml'

name_model = 'EEI_model_5s_fi'

batch = 8  ## img 640 일때는 32로 함
epochs = 500
source = 'train'

test_data_path = copy_data_path +'/images/test'
# val_data_path = copy_data_path+'/images/val'
script_path_detect = './yolov5/detect.py'
weight_path = './yolov5/runs/train/EEI_model_5s_fi2/weights/best.pt'
img = 2464   ### 640 2464로하니 메모리 나감
conf = 0.8
iou = 0.45

## 640, 32로 할때 생각 보다 메모리를 안먹어서 늘려서 해도 될듯? 늘림에 따라 iou thres 값도 올려도댐