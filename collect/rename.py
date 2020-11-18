import os

path = '/Users/haoye.tian/Documents/University/data/PatchCollecting/ssFix'
tool = path.split('/')[-1]

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.patch'):
            project = root.split('/')[-1]
            label = root.split('/')[-2]
            id = file.split('.')[0]
            if label == 'Correct':
                new_name = 'patch1-' + project + '-' + id + '-' + tool + '.patch'
            elif label == 'Incorrect':
                new_name = 'patch1-' + project + '-' + id + '-' + tool + '-plausible.patch'
            os.rename(os.path.join(root, file), os.path.join(root, new_name))

        elif file.endswith('.buggy'):
            project = root.split('/')[-1]
            label = root.split('/')[-2]
            id = file.split('.')[0]
            if label == 'Correct':
                new_name = 'patch1-' + project + '-' + id + '-' + tool + '.buggy'
            elif label == 'Incorrect':
                new_name = 'patch1-' + project + '-' + id + '-' + tool + '-plausible.buggy'
            os.rename(os.path.join(root, file), os.path.join(root, new_name))

        elif file.endswith('.fixed'):
            project = root.split('/')[-1]
            label = root.split('/')[-2]
            id = file.split('.')[0]
            if label == 'Correct':
                new_name = 'patch1-' + project + '-' + id + '-' + tool + '.fixed'
            elif label == 'Incorrect':
                new_name = 'patch1-' + project + '-' + id + '-' + tool + '-plausible.fixed'
            os.rename(os.path.join(root, file), os.path.join(root, new_name))