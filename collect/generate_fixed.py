import os
import shutil
from subprocess import *

path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1/'

def copy_buggy(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('-s.java'):
                tool = root.split('/')[-4]
                if tool == 'PraPR':
                    continue
                new_file = file.replace('-s.java','-t.java')
                shutil.copy(os.path.join(root, file), os.path.join(root, new_file))


def patching(path):
    total = 0
    generated = 0
    Exp = ''
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                tool = root.split('/')[-4]
                if tool == 'PraPR':
                    continue
                total += 1

                name = file.split('.')[0]

                fixed = name + '-t.java'

                path_patch = os.path.join(root, file)
                path_fixed = os.path.join(root, fixed)

                cmd = 'patch -p0 {} {}'.format(path_fixed, path_patch)
                try:
                    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                        output, errors = p.communicate(timeout=300)
                        # print(output)
                        if errors:
                            raise CalledProcessError(errors, '-1')
                        if 'FAILED' in output:
                            Exp += 'Failed name: {}\n'.format(name, )
                            continue
                except Exception as e:
                    Exp += 'Exception name: {}\n'.format(name, )
                    continue

                generated += 1
                print('generated: {}, new: {}'.format(generated, name))
    print(Exp)
    print('total: {}, generated: {}'.format(total, generated))

if __name__ == '__main__':
    # path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1/GenProgA'
    copy_buggy(path)
    patching(path)