import os
import shutil

path = '/Users/haoye.tian/Documents/University/data/PatchCollectingTOSEM'
path2 = '/Users/haoye.tian/Documents/University/data/PatchCollectingTOSEM2'

def add_folder(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.')[0]

                buggy = name + '-s.java'
                fixed = name + '-t.java'

                folder_name = name.split('-')[0]

                new_root = root.replace('PatchCollectingTOSEM', 'PatchCollectingTOSEM2') + '/' + folder_name
                if not os.path.exists(new_root):
                    os.makedirs(new_root)

                if os.path.exists(os.path.join(root, buggy)):
                    shutil.copy(os.path.join(root, buggy), os.path.join(new_root, buggy))
                    shutil.copy(os.path.join(root, fixed), os.path.join(new_root, fixed))
                shutil.copy(os.path.join(root, file), os.path.join(new_root, file))

def detect_patch_snippets(path2):
    cnt = 0
    for root, dirs, files in os.walk(path2):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.')[0]
                patch_id= name.split('-')[0]
                if '_'  in patch_id:
                    print(root)
                    cnt += 1
    print(cnt)

add_folder(path)
# detect_patch_snippets(path2)