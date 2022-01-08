import os
import json
import shutil


def format4Underscore(path, project):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.')[0]

                patch = file
                buggy = name + '-s.java'
                fixed = name + '-t.java'

                head = name.split('-')[0]
                if '_' in head:
                    new_head = head.replace('_', '#')
                    name = name.replace(head, new_head)
                    # patch = patch.replace(head, new_head)
                    # buggy = buggy.replace(head, new_head)
                    # fixed = fixed.replace(head, new_head)

                frag = name.split('-')
                half1 = '-'.join([frag[0], frag[1], frag[2]])

                new_patch = '_'.join([half1, frag[3]]) + '.patch'
                new_buggy = '_'.join([half1, frag[3]]) + '_s.java'
                new_fixed = '_'.join([half1, frag[3]]) + '_t.java'

                folder1 = new_buggy.split('_')[0]
                folder2 = new_buggy.split('_')[1]
                middle = '_'.join([folder1, folder2])

                root_frag = root.split('/')
                new_root = '/'.join(root_frag[:-1]) + '/' + middle + '/' + folder1 + '/' + folder2
                new_root = new_root.replace(project, project +'2')
                if not os.path.exists(new_root):
                    os.makedirs(new_root)

                try:
                    if os.path.exists(os.path.join(root, buggy)):
                        shutil.copy(os.path.join(root, buggy), os.path.join(new_root, new_buggy))
                    if os.path.exists(os.path.join(root, fixed)):
                        shutil.copy(os.path.join(root, fixed), os.path.join(new_root, new_fixed))
                    shutil.copy(os.path.join(root, patch), os.path.join(new_root, new_patch))
                except Exception as e:
                    print(e)
                    continue



# path = '/Users/haoye.tian/Documents/University/data/PatchCollectingTOSEMYe'
path = '/Users/haoye.tian/Documents/University/data/Develop_standardize_sliced_part'
project = 'Develop_standardize_sliced_part'
format4Underscore(path, project)