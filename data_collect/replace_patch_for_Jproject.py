import os
from subprocess import *

J_projects = ['jKali', 'jMutRepair', 'Cardumen']

path = '/Users/haoye.tian/Documents/University/data/PatchCollecting/'

for tool in J_projects:
    for root, dirs, files in os.walk(os.path.join(path, tool)):
        for file in files:
            if file.endswith('.buggy'):
                buggy = os.path.join(root, file)
                fixed = os.path.join(root, file.split('.')[0] + '.fixed')
                patch = os.path.join(root, file.split('.')[0] + '.patch')

                diff_cmd = 'diff -ub {} {} > {}'.format(buggy, fixed, patch)

                try:
                    with Popen(diff_cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                        output, errors = p.communicate(timeout=300)
                        # print(output)
                        if errors:
                            raise CalledProcessError(errors, '-1')
                except Exception as e:
                    print(e)