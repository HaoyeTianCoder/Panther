import os

# path = '/Users/haoye.tian/Downloads/PatchesAndScript/Patches/Dcorrect/DeepRepair/Math'
# path = '/Users/haoye.tian/Downloads/PatchesAndScript'
path = '/Users/haoye.tian/Documents/University/data/PatchCollectingTOSEM/'
# path = '/Users/haoye.tian/Downloads/PatchCollecting/'

def split(path):
    cnt = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path) as f:
                        difffiles = f.read().split('\n\n\n')
                        if len(difffiles) > 1 and len(difffiles[1]) >= 5:
                            file_origin = os.path.join(root, file)
                            for i in range(len(difffiles)):
                                diff = difffiles[i]
                                if len(diff) < 5:
                                    continue
                                cnt += 1
                                print('patch having multi fixes: {}'.format(file))
                                if not diff.endswith('\n'):
                                    diff += '\n'
                                new_list = file_origin.split('-')
                                new_list.insert(1, str(i+1))
                                file_new = new_list[0] + '_' + new_list[1] + '-' + '-'.join(new_list[2:])
                                f = open(file_new,'w')
                                f.write(diff)
                                f.close()
                            os.remove(file_path)
                except Exception as e:
                    print(e)
    print(cnt)

def check(path):
    unique_patch = set()
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):

                if '_' in root:
                    key = root.split('_')[0]
                else:
                    key = root
                unique_patch.add(key)

                file_path = os.path.join(root, file)
                try:
                    cnt = 0
                    with open(file_path, 'r+') as f:
                        for line in f:
                            if line.startswith('---'):
                                cnt += 1
                    if cnt > 1:
                        raise
                except Exception as e:
                    print(e)
    print(len(unique_patch))

# split(path)
check(path)