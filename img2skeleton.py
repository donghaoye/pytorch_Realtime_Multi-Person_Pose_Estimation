# -*- coding: UTF-8 -*-

import os
from picture_demo import img2skeleton


for fpathe,dirs,fs in os.walk('/data/donghaoye/KTH/data/TRAIN'):
    ske_dir = fpathe + "_skeleton"
    if os.path.exists(ske_dir) == False:
        os.makedirs(ske_dir)
    for f in fs:
        img_file = os.path.join(fpathe,f)
        ske_file = os.path.join(ske_dir,"ske_" + f)
        print(img_file)
        print(ske_file)
        img2skeleton(img_file, ske_file)

print ("------------------------END------------------------")