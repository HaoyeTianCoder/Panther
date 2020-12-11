import os
import shutil

path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1/DynaMoth/'

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.patch'):
            name = file.split('.')[0]
            buggy = name + '-s.java'
            fixed = name + '-t.java'

            id = name.split('-')[2]
            if id in root:
                continue
            new_root = os.path.join(root, id)
            if not os.path.exists(new_root):
                os.makedirs(new_root)

            shutil.copy(os.path.join(root, buggy), os.path.join(new_root, buggy))
            shutil.copy(os.path.join(root, fixed), os.path.join(new_root, fixed))
            shutil.copy(os.path.join(root, file), os.path.join(new_root, file))
