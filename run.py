import cv2

from src.lib.detectors.detector_factory import detector_factory
from src.lib.opts import opts

MODEL_PATH = f'./model_last.pth'
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {} --dataset HT --arch resdcn_18 --debug 0 --gpus 16 --input_h 410 --input_w 720'.format(TASK, MODEL_PATH).split(' '))
print(opt)
opt.heads['hm']=1
detector = detector_factory[opt.task](opt)

wrktm = 0
it = 1
supit = 0
limits = 198
with open('/content/temp', 'w+') as f:
  for i in range(limits):
    img = cv2.imread(f'/content/CenterNet/data/HT/test_HT21/{i}.jpg')
    retres18 = detector.run(img)
    wrktm += retres18['net']
    it += 1
print(wrktm/2871)