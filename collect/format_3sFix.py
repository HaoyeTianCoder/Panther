import os
import shutil
from subprocess import *

path = '/Users/haoye.tian/Documents/University/data/PatchCollecting/3sFix/UNknow'

def format():
    projects = os.listdir(path)
    for project in projects:
        if project == '.DS_Store':
            continue
        project_path = os.path.join(path, project)
        bug_ids = os.listdir(project_path)
        for bug_id in bug_ids:
            if not '_' in bug_id or bug_id.startswith('.'):
                continue
            bug_id_path = os.path.join(project_path, bug_id)
            patchids = os.listdir(bug_id_path)
            if 'stdout.log.full' in patchids:
                patchids.remove('stdout.log.full')
            for i in range(len(patchids)):
                patchid = patchids[i]
                patchid_path = os.path.join(bug_id_path, patchid)
                diff_fixed = os.listdir(patchid_path)
                for f in diff_fixed:
                    new_path = project_path
                    if f == 'diff':
                        new_name = 'patch' + str(i+1) + '-' + project + '-' + bug_id.split('_')[1] + '-3sFix.patch'
                        shutil.copyfile(os.path.join(patchid_path, f), os.path.join(new_path, new_name))
                    else:
                        new_name = 'patch' + str(i+1) + '-' + project + '-' + bug_id.split('_')[1] + '-3sFix.fixed'
                        shutil.copyfile(os.path.join(patchid_path, f), os.path.join(new_path, new_name))

def obtain_buggy():
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.')[0]
                path_patch = os.path.join(root, file)
                path_fixed = os.path.join(root, name+'.fixed')
                path_buggy = os.path.join(root, name+'.buggy')
                new_patch = ''
                with open(path_patch,'r') as f:
                    for line in f:
                        if line.startswith('--- '):
                            new_patch += '--- ' + name+'.buggy' + '\n'
                        elif line.startswith('+++ '):
                            new_patch += '+++ ' + name+'.buggy' + '\n'
                        elif line.startswith('-'):
                            new_patch += '+' + line[1:]
                        elif line.startswith('+'):
                            new_patch += '-' + line[1:]
                        else:
                            new_patch += line

                shutil.copyfile(path_fixed,path_buggy)
                with open('/tmp/tmp.patch','w+') as f:
                    f.write(new_patch)

                cmd = 'cd {} && patch --ignore-whitespace -p0 < "/tmp/tmp.patch"'.format(root)
                with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                    output, errors = p.communicate(timeout=300)
                    print(output)
                    if errors:
                        raise CalledProcessError(errors, '-1')


# format()
# obtain_buggy()