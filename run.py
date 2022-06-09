import cv2

import sys
CENTERNET_PATH = '/content/CenterNet/src/lib'
sys.path.insert(0, CENTERNET_PATH)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model_path", help="path to model", type=str)
parser.add_argument("-seq_path", help="path to video seq", type=str)
args = parser.parse_args()

from src.lib.detectors.detector_factory import detector_factory
from src.lib.opts import opts

#MODEL_PATH = f'/content/drive/MyDrive/D_data/model/resdcn18_l/model_45.pth'
MODEL_PATH = args.model_path
DATA_PATH = args.seq_path
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {} --dataset HT --arch resdcn_18 --debug 0 --gpus 16 --input_h 410 --input_w 720'.format(TASK, MODEL_PATH).split(' '))
print(opt)
opt.heads['hm']=1
detector = detector_factory[opt.task](opt)

wrktm = 0
it = 1
while 1:
    img = cv2.imread(DATA_PATH + '/' + ('0' * (6 - len(str(it)))) + str(it) + '.jpg')
    try:
      retres18 = detector.run(img)
    except TypeError:
      break
    wrktm += retres18['net']
    it += 1
print('network time is ', wrktm/(it-1))