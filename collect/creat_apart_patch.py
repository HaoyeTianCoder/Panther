from collect.important.extract_source_file import *

path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1/'

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
                                    new_root = root.replace('PatchCollectingV1', 'PatchCollectingV2')
                                    if not os.path.exists(new_root):
                                        os.makedirs(new_root)
                                    with open(os.path.join(new_root, new_patch_frag), 'w+', ) as p:
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

                    new_root = root.replace('PatchCollectingV1', 'PatchCollectingV2')
                    if not os.path.exists(new_root):
                        os.makedirs(new_root)
                    with open(os.path.join(new_root, new_patch_frag), 'w+',) as p:
                        p.write(patch_str)
                    shutil.copy(os.path.join(root, buggy), os.path.join(new_root, buggy))

def rewrite_patch(path_patch, cmd):
    tmp = '/tmp/tmp.patch'
    shutil.copy(path_patch, tmp)
    for i in range(2):
        original = open(tmp, 'r+', newline='\n')
        lines = original.readlines()
        original.close()

        open(tmp,'w+',newline='\r\n').write(''.join(lines))

    cmd = cmd.replace(path_patch, tmp)
    try:
        with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
            output, errors = p.communicate(timeout=300)
            # print(output)
            if errors:
                raise CalledProcessError(errors, '-1')
            if 'FAILED' in output:
                return 'FAILED'
    except Exception as e:
        return 'FAILED'
    finally:
        os.remove(tmp)
    return 'SUCCESS'

def patching(path):
    total = 0
    generated = 0
    Exp = ''
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                name = file
                if not '-part' in file:
                    continue
                if file == 'patch1-part1-Lang-25-Developer.patch':
                    continue

                total += 1

                l = file.split('.')[0].split('-')
                buggy = '-'.join([l[0], l[2], l[3], l[4]]) + '-s.java'
                new_buggy = '-'.join([l[0], l[1], l[2], l[3], l[4]]) + '-s.java'
                # copy buggy for patch
                shutil.copy(os.path.join(root, buggy), os.path.join(root, new_buggy))

                path_target_buggy = os.path.join(root, new_buggy)
                fixed = file.replace('.patch', '-t.java')
                path_target_fixed = os.path.join(root, fixed)
                # add -t.java
                shutil.copy(path_target_buggy, path_target_fixed)

                tool = file.split('-')[-4]
                project = file.split('-')[-2]
                id = file.split('-')[-1]

                cmd = 'patch -p0 {} {}'.format(path_target_fixed, os.path.join(root, file))
                try:
                    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                        output, errors = p.communicate(timeout=300)
                        # print(output)
                        if errors:
                            raise CalledProcessError(errors, '-1')
                        if 'FAILED' in output:
                            result = rewrite_patch(os.path.join(root, file), cmd)
                            if result == 'FAILED':
                                Exp += 'Failed name: {}\n'.format(name, )
                                continue
                except Exception as e:
                    result = rewrite_patch(os.path.join(root, file), cmd)
                    if result == 'FAILED':
                        Exp += 'Exception name: {}\n'.format(name, )
                        continue

                generated += 1
                print('generated: {}, new: {}'.format(generated, name))

    print(Exp)
    print('total: {}, generated: {}'.format(total, generated))

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

    slice_patch(path)

    new_path = path.replace('PatchCollectingV1','PatchCollectingV2')
    patching(new_path)

    # run_all(new_path)

    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         if file.endswith('.patch'):
    #             if '-part' in file:
    #                 os.remove(os.path.join(root, file))