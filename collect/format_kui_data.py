import os
import shutil

old_path = '/Users/haoye.tian/Downloads/APR-Efficiency-master/Patches/NFL/'
target_path = '/Users/haoye.tian/Documents/University/data/PatchCollecting/'

tools = ['AVATAR', 'FixMiner', 'TBar', 'kPAR', 'jKali', 'jMutRepair', 'Cardumen',
         'GenProgA', 'KaliA', 'RSRepairA']

tools = ['DynaMoth']
# AVATAR = old_path + 'AVATAR'
# FixMiner = old_path + 'FixMiner'
# TBar = old_path + 'TBar'
# kPAR = old_path + 'kPAR'
# jKali = old_path + 'jKali'
# jMutRepair = old_path + 'jMutRepair'
# Cardumen = old_path + 'Cardumen'

dict_tool = {}
for tool in tools:
    dict_tool[tool] = old_path + tool


for tool, path in dict_tool.items():
    print('tool: {}'.format(tool))
    for root, dirs, files in os.walk(path):
        for i in range(len(files)):
            f = files[i]
            if f.endswith('.txt'):
                folder = root.split('/')[-1]
                bug_id = folder.split('_')[0]
                label = folder.split('_')[1]
                project = bug_id.split('-')[0]

                if len(files) > 1:
                    new_name = 'patch1_' + str(i+1) + '-' + bug_id + '-' + tool + '.patch'
                else:
                    new_name = 'patch1-' + bug_id + '-' + tool + '.patch'

                if label == 'C':
                    destination_folder = target_path + tool + '/Correct/' + project + '/'
                elif label == 'P':
                    destination_folder = target_path + tool + '/Incorrect/' + project + '/'
                    # new_name = new_name.replace('.patch', '-plausible.patch')

                destination = destination_folder + new_name

                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                shutil.copyfile(os.path.join(root, f), destination)



