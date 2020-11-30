import os
import shutil
from subprocess import *

path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV2UniqueTokenNewPatch'

def generate_standard_patch(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                patch = os.path.join(root, file)
                name = file.split('.')[0]
                buggy = os.path.join(root, name+ '-s.java')
                fixed = os.path.join(root, name + '-t.java')

                diff_cmd = 'diff -ub {} {} > {}'.format(buggy, fixed, patch)

                try:
                    with Popen(diff_cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                        output, errors = p.communicate(timeout=300)
                        # print(output)
                        if errors:
                            raise CalledProcessError(errors, '-1')
                except Exception as e:
                    print(e)

generate_standard_patch(path)