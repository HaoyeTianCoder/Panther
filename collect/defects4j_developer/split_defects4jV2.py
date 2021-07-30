import os

def slice_patch2(path, folder, new_folder):
    cnt = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                id = root.split('/')[-2]
                project = root.split('/')[-3]
                name = file.split('.')[0]
                number_diff = 0
                number_AT = 0
                patch = ''
                content = False
                try:
                    with open(os.path.join(root, file)) as f:
                        for line in f:
                            if line.startswith('diff') or line.startswith('index'):
                                continue
                            if line.startswith('--- ') or line.startswith('-- '):
                                if number_diff > 0:
                                    # save previous patch
                                    new_path = root.replace(folder, new_folder)

                                    new_list = name.split('-')
                                    new_list.insert(1, str(number_diff))
                                    new_name = new_list[0] + '_' + new_list[1] + '-' + '-'.join(new_list[2:]) +'.patch'

                                    if not os.path.exists(new_path):
                                        os.makedirs(new_path)
                                    with open(os.path.join(new_path, new_name), 'w+') as f:
                                        f.write(minus_line + plus_line + patch)

                                    content = False

                                number_diff += 1
                                minus_line = line
                            elif line.startswith('+++ ') or line.startswith('++ '):
                                plus_line = line
                                content = True
                                patch = ''

                            # elif line.startswith('@@ '):
                            #     if content:
                            #         # save previous patch
                            #         new_path = root.replace(folder, new_folder)
                            #         new_name = name+'-'+str(number_AT)+'.patch'
                            #         if not os.path.exists(new_path):
                            #             os.makedirs(new_path)
                            #         with open(os.path.join(new_path, new_name), 'w+') as f:
                            #             f.write(minus_line + plus_line + patch)
                            #
                            #         patch = line
                            #         number_AT += 1
                            #         content = True
                            #     else:
                            #         # first @@
                            #         patch = line
                            #         number_AT += 1
                            #         content = True

                            elif content:
                                patch += line
                            else:
                                continue
                except Exception as e:
                    print(e)

                # save last patch
                new_path = root.replace(folder, new_folder)

                # new_name = name + '-' + str(number_diff) + '.patch'
                new_list = name.split('-')
                new_list.insert(1, str(number_diff))
                new_name = new_list[0] + '_' + new_list[1] + '-' + '-'.join(new_list[2:]) + '.patch'

                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                with open(os.path.join(new_path, new_name), 'w+') as f:
                    f.write(minus_line + plus_line + patch)

# split patch into new patch carrying with one file changed(possiblely multiple fixes)

# path = '/Users/haoye.tian/Documents/University/data/otherDeve'
path = '/Users/haoye.tian/Documents/University/data/Develop_standardize'
folder = 'Develop_standardize'
new_folder = folder + '_sliced_part'
slice_patch2(path, folder, new_folder)