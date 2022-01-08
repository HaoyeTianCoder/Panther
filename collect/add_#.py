import os

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
path = '/Users/haoye.tian/Documents/University/data/PatchCollectingTOSEMYe/Developer/Correct'
def fix_name(path):
    for project in projects:
        new_path = os.path.join(path, project)
        for root, dirs, files in os.walk(new_path):
            for file in files:
                if not '#' in file and file.endswith('.patch'):
                    name = file.split('.')[0]
                    i = name.index('-')
                    new_name = name[:i] + '#1' + name[i:]

                    old_patch = os.path.join(root, name+'.patch')
                    old_buggy = os.path.join(root, name+'_s.java')
                    old_fixed = os.path.join(root, name+'_t.java')

                    new_patch = os.path.join(root, new_name+'.patch')
                    new_buggy = os.path.join(root, new_name+'_s.java')
                    new_fixed = os.path.join(root, new_name+'_t.java')

                    # fix file name
                    os.rename(old_patch, new_patch)
                    os.rename(old_buggy, new_buggy)
                    os.rename(old_fixed, new_fixed)

                    middle = new_name
                    folder1 = new_name.split('_')[0]
                    folder2 = new_name.split('_')[1]

                    path_frag = root.split('/')

                    # fix folder1 name
                    path_list1 = path_frag[:-1]
                    old_root1 = '/'.join(path_list1)
                    path_list1[-1] = folder1
                    new_root1 = '/'.join(path_list1)
                    os.rename(old_root1, new_root1)

                    # fix middle folder name
                    path_list2 = path_list1[:-1]
                    old_root2 = '/'.join(path_list2)
                    path_list2[-1] = middle
                    new_root2 = '/'.join(path_list2)
                    os.rename(old_root2, new_root2)

# add # for name of path and patch
fix_name(path)