from config_default_old import *
import os, shutil
from subprocess import *

kui_tools = ['AVATAR', 'FixMiner', 'TBar', 'kPAR','iFixR','DynaMoth']
J_projects = ['jKali', 'jMutRepair', 'Cardumen']
A_projects = ['GenProgA', 'KaliA', 'RSRepairA']

def extract_source_file(defects4j_buggy, path_patch, tools):
    for tool in tools:
        patch_tool = os.path.join(path_patch,tool)

def extract4normal(tool):
    cnt = 0
    path_tool = os.path.join(path_patch, tool)
    for root, dirs, files in os.walk(path_tool):
        for file in files:
            if not file.endswith('.patch'):
                continue
            print('processing {}: {}'.format(cnt, file))

            project = file.split('-')[1]
            id = file.split('-')[2]
            path_target_buggy = os.path.join(root, file.replace('.patch', '.buggy'))
            path_target_fixed = os.path.join(root, file.replace('.patch', '.fixed'))

            # obtain buggy file
            obtain_buggy(root, file, project, id, path_target_buggy, tool)

            # parse patch
            fix_operation = parse_patch(root, file, project, id, tool)

            # obtain fixed
            fixed_final_file = obtain_fixed(root, file, project, id, fix_operation, path_target_buggy)

            # save to fixed file
            save_fixed(fixed_final_file, path_target_fixed)

            cnt += 1

# not used
def extract4ssFix1():
    tool = 'ssFix'
    cnt = 0
    path_tool = os.path.join(path_patch, tool)
    for root, dirs, files in os.walk(path_tool):
        for file in files:
            if file.startswith('.'):
                continue
            bug = file.split('_')[0]
            id = file.split('_')[1]
            old = os.path.join(root,file)
            new = os.path.join(root, bug)
            if not os.path.exists(new):
                os.makedirs(new)
            shutil.move(old, os.path.join(new, id+'.src.patch'))
            cnt += 1

def extract4ssFix2():
    tool = 'ssFix'
    cnt = 0
    path_tool = os.path.join(path_patch, tool)
    for root, dirs, files in os.walk(path_tool):
        for file in files:
            if not file.endswith('.patch'):
                continue
            project = file.split('-')[1]
            id = file.split('-')[2]
            path_target_buggy = os.path.join(root, file.replace('.patch', '.buggy'))
            path_target_fixed = os.path.join(root, file.replace('.patch', '.fixed'))

            # obtain buggy file
            obtain_buggy(root, file, project, id, path_target_buggy, tool)

            # parse patch
            fix_operation = parse_patch(root, file, project, id, tool)

            # obtain fixed
            fixed_final_file = obtain_fixed(root, file, project, id, fix_operation, path_target_buggy)

            # save to fixed file
            save_fixed(fixed_final_file, path_target_fixed)

            cnt += 1
            print('{} finished...'.format(cnt))

def obtain_buggy(root, file, project, id, path_target_buggy, tool):
    # obtain buggy file
    path_buggy_file = ''

    if tool in J_projects:
        with open(os.path.join(root, file), 'r+') as f:
            for line in f:
                if line.startswith('++'):
                        path_line = line.split(' ')[1].strip()
                        discrete = path_line.split('/')
                        path_buggy_file = '/' + '/'.join(discrete[3:])
                        path_buggy_file = path_buggy_file[:-4] + '.java'
    else:
        with open(os.path.join(root, file), 'r+') as f:
            for line in f:
                if line.startswith('--'):
                    if tool in kui_tools:
                        path_buggy_file = line.split(' ')[1].strip()[1:]
                    elif tool in A_projects:
                        path_line = line.split(' ')[1].strip()
                        path_line = path_line.split('\t')[0]
                        discrete = path_line.split('/')
                        path_buggy_file = '/' + '/'.join(discrete[3:])
                    else:
                        path_buggy_file = line.split(' ')[1].strip()
                    if tool == 'DeepRepair':
                        path_buggy_file = path_buggy_file.replace('//','/')
                    break
    if path_buggy_file == '':
        raise
    bug_id = project + '_' + id
    path_buggy_file_repo = os.path.join(defects4j_buggy, bug_id) + path_buggy_file

    shutil.copyfile(path_buggy_file_repo, path_target_buggy)

def obtain_buggy_without_src(root, file, project, id):
    # obtain buggy file
    with open(os.path.join(root, file), 'r+') as f:
        for line in f:
            if line.startswith('++'):
                path_buggy_file = line.split(' ')[1].strip()
                break
    bug_id = project + '_' + id
    if project == 'Chart':
        path_buggy_file_repo = os.path.join(defects4j_buggy, bug_id, 'source', path_buggy_file)
    elif project == 'Closure' or project == 'Mockito':
        path_buggy_file_repo = os.path.join(defects4j_buggy, bug_id, 'src', path_buggy_file)
    else:
        path_buggy_file_repo = os.path.join(defects4j_buggy, bug_id, 'src', path_buggy_file)
        if not os.path.exists(path_buggy_file_repo):
            path_buggy_file_repo = os.path.join(defects4j_buggy, bug_id, 'src/main/java', path_buggy_file)
        elif not os.path.exists(path_buggy_file_repo):
            path_buggy_file_repo = os.path.join(defects4j_buggy, bug_id, 'src/java', path_buggy_file)
    path_target = os.path.join(root, id + '.buggy')
    path_target_fixed = os.path.join(root, id + '.fixed')
    shutil.copyfile(path_buggy_file_repo, path_target)

def parse_patch(root, file, project, id, tool):

    # parse patch operation
    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
        found = False
        fix_operation = []
        buggy_lines = 0
        fixed_chunk = []
        nubmer_at = 0
        diff = 0
        for line in f:
            # exclude specific extra line
            if line.startswith('\ No newline at end of file') or line.startswith('Common subdirectories'):
                    # or line.startswith('diff') or line.startswith('index'):
                continue
            if line == '\n':
                if tool == 'PraPR':
                    continue
            if not found and not line.startswith('@@ '):
                continue
            elif line.startswith('@@ '):
                found = True

                if nubmer_at == 1:
                    # fix_operation.append([bug_location_line, buggy_lines, fixed_chunk+['\n']])
                    if not fixed_chunk[-1].endswith('\n'):
                        fixed_chunk[-1] += '\n'
                    fix_operation.append([bug_location_line, buggy_lines, fixed_chunk])

                if nubmer_at > 1:
                    # handle next @@ fix location influenced by last @@ fix
                    old_hunk_length = fix_operation[-1][1]
                    new_hunk_length = len(fix_operation[-1][2])
                    diff += new_hunk_length - old_hunk_length
                    bug_location_line += diff

                    # fix_operation.append([bug_location_line, buggy_lines, fixed_chunk+['\n']])
                    if not fixed_chunk[-1].endswith('\n'):
                        fixed_chunk[-1] += '\n'
                    fix_operation.append([bug_location_line, buggy_lines, fixed_chunk])

                nubmer_at += 1
                # reset, next @@ fix location
                fixed_chunk = []
                # buggy_lines = 0
                # bug_location_line = int(line.split(' ')[1].split(',')[0][1:])

                # v2: record buggy location and lines
                num_with_space = line.split('-')[1]
                num = num_with_space.split(' ')[0]
                bug_location_line, buggy_lines = int(num.split(',')[0]), int(num.split(',')[1])
            else:
                if line.startswith('-'):
                    # buggy_lines += 1
                    pass
                elif line.startswith('+'):
                    fixed_chunk.append(line[1:])
                else:
                    # buggy_lines += 1
                    pass
                    fixed_chunk.append(line)

        # handle last @@
        if nubmer_at > 1:
            # handle next @@ fix location influenced by last @@ fix
            old_hunk_length = fix_operation[-1][1]
            new_hunk_length = len(fix_operation[-1][2])
            diff += new_hunk_length - old_hunk_length
            bug_location_line += diff

        # fix_operation.append([bug_location_line, buggy_lines, fixed_chunk+['\n']])
        if not fixed_chunk[-1].endswith('\n'):
            fixed_chunk[-1] += '\n'
        fix_operation.append([bug_location_line, buggy_lines, fixed_chunk])

    return fix_operation

def obtain_fixed(root, file, project, id, fix_operation, path_target):
    # obtain fixed file
    fixed_final_file = ''
    with open(path_target, 'r+') as f:
        try:
            fixed_final_file = f.readlines()
        except Exception:
            print('error: {}'.format(file))
            return ['null file']
        # fixed_final_file = [line for line in f]
        for i in range(len(fix_operation)):
            bug_location_line, buggy_lines, fixed_chunk = fix_operation[i][0], fix_operation[i][1], fix_operation[i][2]

            # handle extra space at the end of chunk
            while fixed_chunk[-1] == '\n' and fixed_chunk[-2].endswith('\n'):
                del fixed_chunk[-1]

            # handle different project
            pass

            head = fixed_final_file[:bug_location_line - 1]
            tail = fixed_final_file[bug_location_line - 1 + buggy_lines:]
            fixed_final_file = head + fixed_chunk + tail
    return fixed_final_file

def save_fixed(fixed_final_file, path_target_fixed):
    fixed_final_file = ''.join(fixed_final_file)
    with open(path_target_fixed, 'w+') as f:
        f.write(fixed_final_file)

def obtain_fixed_4PraPR(root, file, project, id, fix_operation, path_target_buggy):
    fixed_final_file = ''
    with open(path_target_buggy, 'r') as f:
        fixed_final_file = f.readlines()
        for i in range(len(fix_operation)):
            bug_location_line, buggy_lines, fixed_chunk = fix_operation[i][0], fix_operation[i][1], \
                                                          fix_operation[i][2]
            # handle different project
            if project == 'Chart':
                if id == '20':
                    pass
                else:
                    bug_location_line -= 1

            elif project == 'Closure':
                if id == '10':
                    pass
                elif id == '14':
                    bug_location_line += 2
                elif id == '70':
                    bug_location_line += 1
                elif id == '86':
                    bug_location_line -= 4
                else:
                    bug_location_line -= 1

                if id == '63':
                    buggy_lines += 1

            elif project == 'Lang':
                if id == '10':
                    bug_location_line += 5
                elif id == '26':
                    bug_location_line += 2
                else:
                    bug_location_line -= 1

            elif project == 'Math':
                if id == '70':
                    bug_location_line -= 2
                if id == '85':
                    pass
                else:
                    bug_location_line -= 1

            elif project == 'Mockito':
                bug_location_line -= 1

            elif project == 'Time':
                bug_location_line -= 1

            head = fixed_final_file[:bug_location_line - 1]
            tail = fixed_final_file[bug_location_line - 1 + buggy_lines:]
            fixed_final_file = head + fixed_chunk + tail
    return fixed_final_file

def extract4PraPR():
    tool = 'PraPR'
    cnt = 0
    path_tool = os.path.join(path_patch, tool)
    for root, dirs, files in os.walk(path_tool):
        for file in files:
            if not file.endswith('.patch'):
                continue

            id = str(file.split('.')[0])
            path_target_buggy = os.path.join(root, id + '.buggy')
            path_target_fixed = os.path.join(root, id + '.fixed')
            project = str(root.split('/')[-1])

            # obtain buggy file
            obtain_buggy_without_src(root, file, project, id)

            # parse patch
            fix_operation = parse_patch(root, file, project, id, tool)

            # obtain fixed
            fixed_final_file = obtain_fixed_4PraPR(root, file, project, id, fix_operation, path_target_buggy)


            # save to fixed file
            save_fixed(fixed_final_file, path_target_fixed, )


            # # method 2
            # with open(os.path.join(root,file), "r", encoding="utf-8") as f:
            #     found = False
            #     lines = '--- ' + path_target + '\n'
            #     lines += '+++ ' + path_target.replace('.buggy','.fixed') + '\n'
            #     for line in f:
            #         if not found and not line.startswith('@@ '):
            #             continue
            #         else:
            #             found = True
            #             lines += line
            # with open('/tmp/tmp.patch', "w", encoding="utf-8") as f:
            #     f.write(lines)
            # cmd = 'patch -p0 < /tmp/tmp.patch'
            # with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
            #     output, errors = p.communicate(timeout=300)
            #     # print(output)
            #     if errors:
            #         raise CalledProcessError(errors, '-1')

if __name__ == '__main__':
    # config
    cfg = Config()
    defects4j_buggy = cfg.defects4j_buggy
    path_patch = cfg.path_patch
    tools = cfg.tools

    # extract_source_file(defects4j_buggy, path_patch, tools)

    # extract4PraPR()

    # extract4ssFix2()

    # extract4normal(tool = 'DeepRepair')

    # extract4normal(tool = 'ACS')

    # extract4normal(tool = 'Arja')

    # extract4normal(tool = 'CapGen')

    # extract4normal(tool = 'Elixir')

    # extract4normal(tool = 'HDRepair')

    # extract4normal(tool = 'Jaid')

    # extract4normal(tool = 'JGenProg2015')

    # extract4normal(tool = 'Nopol2015')

    # extract4normal(tool = 'SequenceR')

    # extract4normal(tool = 'SimFix')

    # extract4normal(tool = 'SketchFix')

    # extract4normal(tool = 'SOFix')

    # kui's tools
    # for tool in kui_tools:
    #     extract4normal(tool = tool)

    # J projects
    # for tool in J_projects:
    #     extract4normal(tool = tool)

    # A projects
    # for tool in A_projects:
    #     extract4normal(tool=tool)

    # extract4normal(tool='iFixR')

    # extract4normal(tool='DynaMoth')