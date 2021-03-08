import numpy as np
import json
import random
import os
import shutil

path = '/Users/haoye.tian/Documents/University/data/PatchCollectingTOSEM'

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.patch'):
            name = file.split('.')[0]
            buggy_old = name + '.buggy'
            fixed_old = name + '.fixed'

            buggy_new = name + '-s.java'
            fixed_new = name + '-t.java'

            new_path = root.replace('PatchCollectingTOSEM','PatchCollectingTOSEM2')
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            shutil.copy(os.path.join(root, buggy_old), os.path.join(new_path, buggy_new))
            shutil.copy(os.path.join(root, fixed_old), os.path.join(new_path, fixed_new))
            shutil.copy(os.path.join(root, file), os.path.join(new_path, file))






