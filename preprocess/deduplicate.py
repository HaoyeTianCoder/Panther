import os
import shutil
from subprocess import *
from nltk.tokenize import word_tokenize


path = '/Users/haoye.tian/Documents/University/data/PatchCollecting/'
# v2 contains ods feature
path2 = '/Users/haoye.tian/Documents/University/data/PatchCollectingV2/'

def deduplicate_by_content_with_location(path):
    unique_dict = {}
    pre = exception = post = repeat = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                path_patch = os.path.join(root, file)
                unique_str = ''
                pre += 1
                print('{}, name: {}'.format(pre, file.split('.')[0]))
                try:
                    with open(path_patch, 'r') as f:
                        foundAT = False
                        for line in f:
                            if line.startswith('--') or line.startswith('++'):
                                continue
                            if not foundAT and not line.startswith('@@ '):
                                continue
                            else:
                                if line.startswith('@@ '):
                                    foundAT = True
                                    unique_str += line.strip() + ' '
                                elif line.startswith('-') or line.startswith('+'):
                                    unique_str += line[1:].strip() + ' '
                                else:
                                    unique_str += line.strip() + ' '
                except Exception as e:
                    print('Exception: {}'.format(e))
                    exception += 1
                    continue

                if unique_str in unique_dict:
                    unique_dict[unique_str] += 1
                    repeat += 1
                    continue
                else:
                    unique_dict[unique_str] = 0

                    # copy unique to another folder
                    name = file.split('.')[0]
                    new_path = root.replace('PatchCollectingV2', 'PatchCollectingV2Unique')
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)

                    patch = file
                    buggy = name + '-s.java'
                    fixed = name + '-t.java'
                    feature = 'features_' + buggy + '->' + fixed + '.json'

                    shutil.copy(os.path.join(root, patch), os.path.join(new_path, patch))
                    shutil.copy(os.path.join(root, buggy), os.path.join(new_path, buggy))
                    shutil.copy(os.path.join(root, fixed), os.path.join(new_path, fixed))
                    if os.path.exists(os.path.join(root, feature)):
                        shutil.copy(os.path.join(root, feature), os.path.join(new_path, feature))

                    post += 1

    print('pre:{}, post:{} ---- exception:{}, repeat:{}'.format(pre, post, exception, repeat))


def deduplicate_by_token_with_location(path):
    unique_dict = {}
    pre = exception = post = repeat = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                path_patch = os.path.join(root, file)
                unique_str = ''
                pre += 1
                print('{}, name: {}'.format(pre, file.split('.')[0]))
                try:
                    with open(path_patch, 'r') as f:
                        foundAT = False
                        for line in f:
                            if line.startswith('--') or line.startswith('++'):
                                continue
                            if not foundAT and not line.startswith('@@ '):
                                continue
                            else:
                                if line.startswith('@@ '):
                                    foundAT = True
                                    unique_str += ' '.join(word_tokenize(line.strip())) + ' '
                                elif line.startswith('-') or line.startswith('+'):
                                    unique_str += ' '.join(word_tokenize(line[1:].strip())) + ' '
                                else:
                                    unique_str += ' '.join(word_tokenize(line.strip())) + ' '
                except Exception as e:
                    print('Exception: {}'.format(e))
                    exception += 1
                    continue

                if unique_str in unique_dict:
                    unique_dict[unique_str] += 1
                    repeat += 1
                    continue
                else:
                    unique_dict[unique_str] = 0

                    # copy unique to another folder
                    name = file.split('.')[0]
                    new_path = root.replace('PatchCollectingV2', 'PatchCollectingV2UniqueToken')
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)

                    patch = file
                    buggy = name + '-s.java'
                    fixed = name + '-t.java'
                    feature = 'features_' + buggy + '->' + fixed + '.json'

                    shutil.copy(os.path.join(root, patch), os.path.join(new_path, patch))
                    shutil.copy(os.path.join(root, buggy), os.path.join(new_path, buggy))
                    shutil.copy(os.path.join(root, fixed), os.path.join(new_path, fixed))
                    if os.path.exists(os.path.join(root, feature)):
                        shutil.copy(os.path.join(root, feature), os.path.join(new_path, feature))

                    post += 1

    print('pre:{}, post:{} ---- exception:{}, repeat:{}'.format(pre, post, exception, repeat))

if __name__ == '__main__':

    # by content with location number
    # deduplicate_by_content_with_location(path2)

    # by token content with location number
    deduplicate_by_token_with_location(path2)