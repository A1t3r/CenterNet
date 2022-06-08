import cv2

import sys
CENTERNET_PATH = '/content/CenterNet/src/lib'
sys.path.insert(0, CENTERNET_PATH)

from src.lib.detectors.detector_factory import detector_factory
from src.lib.opts import opts

MODEL_PATH = f'/content/drive/MyDrive/D_data/model/resdcn18_l/model_45.pth'
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {} --dataset HT --arch resdcn_18 --debug 0 --gpus 16 --input_h 410 --input_w 720'.format(TASK, MODEL_PATH).split(' '))
print(opt)
opt.heads['hm']=1
detector = detector_factory[opt.task](opt)

wrktm = 0
it = 1
supit = 0
limits = 1980
for i in range(limits):
    img = cv2.imread(f'/content/drive/MyDrive/D_data/HT21/HT/test_HT21/{i}.jpg')
    retres18 = detector.run(img)
    wrktm += retres18['net']
    it += 1
print(wrktm/1980)