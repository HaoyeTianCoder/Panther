from config_default import *
import os, shutil

def extract_source_file(defects4j_buggy, path_patch, tools):
    for tool in tools:
        patch_tool = os.path.join(path_patch,tool)

def extract4PraPR():
    tool = 'PraPR'
    path_tool = os.path.join(defects4j_buggy, tool)
    for root, dirs, files in os.walk(path_tool):
        for file in files:
            if not file.endswith('.patch'):
                continue

            # obtain buggy file
            with open(os.path.join(root, file),'r+') as f:
                for line in f:
                    if line.startswith('++'):
                        path_fixed_file = line.split(' ')[1].strip()
                        break
            bug_id = str(root.split('/')[-1]) + '_' + str(file.split('.')[0])
            path_buggy_file = os.path.join(defects4j_buggy, bug_id, 'source', path_fixed_file)
            if not os.path.exists(path_fixed_file):
                print('check......')
                raise
            path_target = os.path.join(root, str(file.split('.')[0])+'.buggy')
            shutil.copyfile(path_buggy_file, path_target)

            # obtain fixed file


if __name__ == '__main__':
    # config
    cfg = Config()
    defects4j_buggy = cfg.defects4j_buggy
    path_patch = cfg.path_patch
    tools = cfg.tools

    extract_source_file(defects4j_buggy, path_patch, tools)

