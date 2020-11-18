import os
import shutil

path = '/Users/haoye.tian/Documents/University/data/PatchCollecting/Developer/Correct/'

project_ids = os.listdir(path)
for project_id in project_ids:
    if project_id.startswith('.') or not '_' in project_id:
        continue
    project_id_path = os.path.join(path, project_id)
    file_folders = os.listdir(project_id_path)
    for i in range(len(file_folders)):
        file_folder = file_folders[i]
        file_folder_path = os.path.join(project_id_path, file_folder)
        targets = os.listdir(file_folder_path)
        for tar in targets:
            project = tar.split('_')[0]
            id = tar.split('_')[1]

            if len(file_folders) == 1:
                if tar.endswith('.patch'):
                    new_name = 'patch1-' + project + '-' + id + '-Developer.patch'
                elif tar.endswith('_s.java'):
                    new_name = 'patch1-' + project + '-' + id + '-Developer.buggy'
                elif tar.endswith('_t.java'):
                    new_name = 'patch1-' + project + '-' + id + '-Developer.fixed'

            elif len(file_folders) > 1:
                if tar.endswith('.patch'):
                    new_name = 'patch1_' + str(i+1) + '-' + project + '-' + id + '-Developer.patch'
                elif tar.endswith('_s.java'):
                    new_name = 'patch1_' + str(i+1) + '-' + project + '-' + id + '-Developer.buggy'
                elif tar.endswith('_t.java'):
                    new_name = 'patch1_' + str(i+1) + '-' + project + '-' + id + '-Developer.fixed'


            old = os.path.join(file_folder_path, tar)
            new_root = path + project + '/' + id + '/'
            if not os.path.exists(new_root):
                os.makedirs(new_root)

            new = new_root + new_name
            shutil.copy(old, new)




# for root, dirs, files in os.walk(path):
#     for file in files:
#         if file.startswith('.'):
#             continue
#         project = file.split('_')[0]
#         id = file.split('_')[1]
#         old = os.path.join(root, file)
#
#         if file.endswith('.patch'):
#             new = path + 'Correct/' + project + '/' + id + '/' + new_name

