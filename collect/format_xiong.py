import os
import json
import shutil

def rename_xiong(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('_s.java'):
                patchid = root.split('/')[-1]
                bugid = root.split('/')[-2]
                project = root.split('/')[-3]

                # new_name = '-'.join([patchid, project, bugid, 'PatchSim']) + '_s.java'
                new_name = '-'.join([patchid, project, bugid]) + '_PatchSim_s.java'

                os.rename(os.path.join(root, file), os.path.join(root, new_name))
            elif file.endswith('_t.java'):
                patchid = root.split('/')[-1]
                bugid = root.split('/')[-2]
                project = root.split('/')[-3]

                # new_name = '-'.join([patchid, project, bugid, 'PatchSim']) + '_t.java'
                new_name = '-'.join([patchid, project, bugid]) + '_PatchSim_t.java'

                os.rename(os.path.join(root, file), os.path.join(root, new_name))
            elif file.endswith('.patch'):
                patchid = root.split('/')[-1]
                bugid = root.split('/')[-2]
                project = root.split('/')[-3]

                # new_name = '-'.join([patchid, project, bugid, 'PatchSim']) + '_t.java'
                new_name = '-'.join([patchid, project, bugid]) + '_PatchSim.patch'

                os.rename(os.path.join(root, file), os.path.join(root, new_name))

def addFolder4feature(path, project):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.')[0]
                buggy = name + '_s.java'
                fixed = name + '_t.java'
                patch = file

                folder1 = buggy.split('_')[0]
                folder2 = buggy.split('_')[1]
                middle = '_'.join([folder1, folder2])

                root_frag = root.split('/')
                new_root = '/'.join(root_frag[:-1]) + '/' + middle + '/' + folder1 + '/' + folder2
                new_root = new_root.replace(project, project +'2')
                if not os.path.exists(new_root):
                    os.makedirs(new_root)

                try:
                    shutil.copy(os.path.join(root, buggy), os.path.join(new_root, buggy))
                    shutil.copy(os.path.join(root, fixed), os.path.join(new_root, fixed))
                    shutil.copy(os.path.join(root, patch), os.path.join(new_root, patch))
                except Exception as e:
                    print(e)
                    continue

path = '/Users/haoye.tian/Documents/University/data/PatchSimTOSEM'
project = 'PatchSimTOSEM'

# rename_xiong(path)
addFolder4feature(path, project)