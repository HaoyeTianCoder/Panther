import os
import shutil

path = '/Users/haoye.tian/Documents/University/data/PatchCollecting/iFixR'

for root, dirs, files in os.walk(path):
    if files[0].startswith('.'):
        continue
    folder = root.split('/')[-1]
    project = folder.split('_')[0]
    id = folder.split('_')[1]
    new_path = root.replace(folder, project)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    if len(files) == 1:
        file = files[0]
        new_name = 'patch1-' + project + '-' + id + '-iFixR.patch'
        shutil.move(os.path.join(root,file), os.path.join(new_path, new_name))
    else:
        for i in range(len(files)):
            new_name = 'patch1_'+ str(i+1) + '-' + project + '-' + id + '-iFixR.patch'
            shutil.move(os.path.join(root, files[i]), os.path.join(new_path, new_name))


