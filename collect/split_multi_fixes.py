import os

# path = '/Users/haoye.tian/Downloads/PatchesAndScript/Patches/Dcorrect/DeepRepair/Math'
# path = '/Users/haoye.tian/Downloads/PatchesAndScript'
path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1/GenProgA'

cnt = 0
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.patch'):
            file_path = os.path.join(root, file)
            with open(file_path) as f:
                difffiles = f.read().split('\n\n\n')
                if len(difffiles) > 1 :
                    cnt += 1
                    print('patch having multi fixes: {}'.format(file))
                    file_origin = os.path.join(root, file)
                    for i in range(len(difffiles)):
                        diff = difffiles[i]
                        if not diff.endswith('\n'):
                            diff += '\n'
                        new_list = file_origin.split('-')
                        new_list.insert(1, str(i+1))
                        file_new = new_list[0] + '_' + new_list[1] + '-' + '-'.join(new_list[2:])
                        f = open(file_new,'w')
                        f.write(diff)
                        f.close()
                    os.remove(file_path)
print(cnt)