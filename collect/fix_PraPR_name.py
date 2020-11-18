import os
import shutil
from subprocess import *

path = '/Users/haoye.tian/Documents/University/data/PatchCollecting/PraPR/'

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.patch'):
            if not 'patch1' in root:
                continue
            id = file.split('-')[4]

            new_name = '-'.join([file.split('-')[0], file.split('-')[1], id, file.split('-')[5]])
            new_name += '.patch'

            new_path = root.replace('patch1', id)

            shutil.move(os.path.join(root, file), os.path.join(new_path, new_name))