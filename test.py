import numpy as np
import json
import random
import os
import shutil
import pickle
from subprocess import *
path = '/Users/haoye.tian/Desktop/20/patch1-Chart-20-Developer.patch'

def patching(path):
    with open(path, 'r+', encoding='utf-8') as f:
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
                        new_patch_frag = '/Users/haoye.tian/Desktop/20/patch1-part1-Chart-20-Developer.patch'
                        new_root = '/Users/haoye.tian/Desktop/20/'
                        if not os.path.exists(new_root):
                            os.makedirs(new_root)
                        with open(os.path.join(new_root, new_patch_frag), 'w+') as p:
                            p.write(patch_str)
                        # found = False
                        at_number += 1
                        patch_str = line
                else:
                    if at_number == 0:
                        at_number += 1
                        found = True
                        patch_str = '+++ \n' + line
        except Exception as e:
            print(e)
        # handle last part
        new_patch_frag = '/Users/haoye.tian/Desktop/20/patch1-part1-Chart-20-Developer.patch'
        new_root = '/Users/haoye.tian/Desktop/20/'
        if not os.path.exists(new_root):
            os.makedirs(new_root)
        with open(os.path.join(new_root, new_patch_frag), 'w+', encoding='utf-8') as p:
            p.write(patch_str)


def paste(path):
    result = ""
    with open(path, 'r+', newline='\r\n', encoding='UTF-8') as f:
        for line in f:
            result += line
    # result = result.replace('\n', '\r\n')
    cmd = 'echo \'{}\' > {}'.format(result, '/Users/haoye.tian/Desktop/20/patch1-part1-Chart-20-Developer.patch')
    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
        output, errors = p.communicate(timeout=300)
        print(output)
    # with open('/Users/haoye.tian/Desktop/20/patch1-part1-Chart-20-Developer.patch','w+') as f:
    #     f.writelines(result)

def test(path):
    original = open(path, 'r+', newline='\n')
    lines = original.readlines()
    original.close()
    open(path, 'w+', newline='\r\n').write(''.join(lines))

    original = open(path, 'r+', newline='\r\n')
    lines = original.readlines()
    original.close()
    open(path, 'w+', newline='\r\n').write(''.join(lines))

    original = open(path, 'r+', newline='\r\n')
    lines = original.readlines()
    print(''.join(lines))

# paste(path)
patching(path)
test(path='/Users/haoye.tian/Desktop/20/patch1-part1-Chart-20-Developer.patch')

