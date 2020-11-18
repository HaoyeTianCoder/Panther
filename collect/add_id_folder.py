import os
import shutil

path = '/Users/haoye.tian/Documents/University/data/PatchCollecting/'

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.buggy') or file.endswith('.fixed') or file.endswith('.patch'):
            id = file.split('-')[2]

            if root.split('/')[-1] == id:
                continue

            old = os.path.join(root, file)
            new_root = os.path.join(root, id)

            if not os.path.exists(new_root):
                os.mkdir(new_root)

            shutil.move(old, new_root)


