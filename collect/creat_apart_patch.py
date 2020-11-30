import os
import shutil
from subprocess import *
from collect.extract_source_file import *

path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV2/'

def slice_patch(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                if '-part' in root:
                    continue
                name = file.split('.')[0]
                buggy = name + '-s.java'
                project = file.split('.')[0].split('-')[1]
                with open(os.path.join(root, file)) as f:
                    found = False
                    at_number = 0
                    try:
                        for line in f:
                            if not found and not line.startswith('@@ '):
                                continue
                            elif found:
                                if not line.startswith('@@ '):
                                    patch_str += line
                                else:
                                    new_patch_frag = file.replace('-'+project, '-part' + str(at_number) + '-' + project)
                                    new_root = root.replace('PatchCollectingV2', 'PatchCollectingV3')
                                    if not os.path.exists(new_root):
                                        os.makedirs(new_root)
                                    with open(os.path.join(new_root, new_patch_frag), 'w') as p:
                                        p.write(patch_str)
                                    shutil.copy(os.path.join(root,buggy), os.path.join(new_root, buggy))
                                    # found = False
                                    at_number += 1
                                    patch_str = line
                            else:
                                if at_number == 0:
                                    at_number += 1
                                    found = True
                                    patch_str = '+++ \n' + line
                    except Exception as e:
                        print(file)
                        continue
                    # handle last part
                    new_patch_frag = file.replace('-' + project, '-part' + str(at_number) + '-' + project)

                    new_root = root.replace('PatchCollectingV2', 'PatchCollectingV3')
                    if not os.path.exists(new_root):
                        os.makedirs(new_root)
                    with open(os.path.join(new_root, new_patch_frag), 'w') as p:
                        p.write(patch_str)
                    shutil.copy(os.path.join(root, buggy), os.path.join(new_root, buggy))


def run_all(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                if not '-part' in file:
                    continue
                if file == 'patch1-part1-Lang-25-Developer.patch':
                    continue
                l = file.split('.')[0].split('-')
                buggy = '-'.join([l[0], l[2], l[3], l[4]]) + '-s.java'
                new_buggy = '-'.join([l[0], l[1], l[2], l[3], l[4]]) + '-s.java'
                # copy buggy for patch
                shutil.copy(os.path.join(root, buggy), os.path.join(root, new_buggy))

                path_target_buggy = os.path.join(root, buggy)
                fixed = file.replace('.patch', '-t.java')
                path_target_fixed = os.path.join(root, fixed)

                tool = file.split('-')[-4]
                project = file.split('-')[-2]
                id = file.split('-')[-1]

                # parse patch
                fix_operation = parse_patch(root, file, '', '', tool)

                # obtain fixed
                if tool == 'PraPR':
                    fixed_final_file = obtain_fixed_4PraPR(root, file, project, id, fix_operation, path_target_buggy)
                else:
                    fixed_final_file = obtain_fixed(root, file, '', '', fix_operation, path_target_buggy)

                # save to fixed file
                save_fixed(fixed_final_file, path_target_fixed)

if __name__ == '__main__':

    # slice_patch(path)

    new_path = path.replace('PatchCollectingV2','PatchCollectingV3')
    run_all(new_path)

    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         if file.endswith('.patch'):
    #             if '-part' in file:
    #                 os.remove(os.path.join(root, file))