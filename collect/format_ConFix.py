import os
import shutil

path = '/Users/haoye.tian/Documents/University/data/PatchCollectingV2UniqueTokenFix/ConFix'

def format():
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == 'confix.diff':
                buggy_folder = os.path.join(root, 'buggy')
                for root2, dirs2, files2 in os.walk(buggy_folder):
                    for file2 in files2:
                        if file2.endswith('.java'):
                            buggy_path = os.path.join(root2, file2)
                            break

                fixed_folder = os.path.join(root, 'confix')
                for root3, dirs3, files3 in os.walk(fixed_folder):
                    for file3 in files3:
                        if file3.endswith('.java') and file3 == file2:
                            fixed_path = os.path.join(root3, file3)
                            break

                patch_path = os.path.join(root, file)

                # format
                project = root.split('/')[-2]
                bugid = root.split('/')[-1]
                if project == 'Closure':
                    id = bugid[7:]
                elif project == 'Chart':
                    id = bugid[5:]
                else:
                    id = bugid[4:]

                pre = root.split('/')[:-1]
                pre = '/'.join(pre)
                new_name = 'patch1-' + project + '-' + id + '-ConFix.'

                shutil.copyfile(buggy_path,os.path.join(pre, new_name+'buggy'))
                shutil.copyfile(fixed_path,os.path.join(pre, new_name+'fixed'))
                shutil.copyfile(patch_path,os.path.join(pre, new_name+'patch'))

corrects = ['chart1','chart10','chart11','chart24','closure14','closure38','closure73','closure92','closure93','closure109','lang6','lang24','lang26','lang51','lang57','math5','math30','math33','math34',\
            'math70','math75','time19',]
incorrects = ['chart3','chart5','chart7','chart9','chart12','chart13','chart15','chart25','chart26','closure2','closure21','closure22','closure46','closure55','closure59','closure79', \
              'closure83','closure89','closure90','closure108','closure119','closure125','closure126','closure133','lang7','lang22','lang27','lang31','lang39','lang43','lang45','lang59', \
              'lang60','lang63','math2','math3','math7','math8','math18','math20','math28','math29','math32','math40','math42','math44','math49','math50','math56','math57','math58','math61',\
              'math62','math63','math74','math78','math79','math80','math81','math82','math84','math85','math88','math94','math95','time4','time7','time9','time11','time17',]
def label():
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.buggy'):
                project = file.split('-')[1]
                project = project.lower()
                id = file.split('-')[2]
                name = file.split('.')[0]
                if root.split('/')[-1] == 'Correct' or root.split('/')[-1] == 'Incorrect':
                    continue
                if project+id in corrects:
                    shutil.copy(os.path.join(root, file), root+'/../Correct')
                    shutil.copy(os.path.join(root, name+'.fixed'), root+'/../Correct')
                    shutil.copy(os.path.join(root, name+'.patch'), root+'/../Correct')
                elif project+id in incorrects:
                    shutil.copy(os.path.join(root, file), root+'/../Incorrect')
                    shutil.copy(os.path.join(root, name+'.fixed'), root+'/../Incorrect')
                    shutil.copy(os.path.join(root, name+'.patch'), root+'/../Incorrect')
                else:
                    raise

def move():
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                project = file.split('-')[1]
                # project = project.lower()
                id = file.split('-')[2]
                name = file.split('.')[0]
                buggy = name + '-s.java'
                fixed = name + '-t.java'

                feature_name = 'features_' + buggy + '->' + fixed + '.json'

                new_path = root + '/../' + project + '/' + id
                if not os.path.exists(new_path):
                    os.makedirs(new_path)

                if not project in root:
                    shutil.copy(os.path.join(root, file), new_path)
                    shutil.copy(os.path.join(root, buggy), new_path)
                    shutil.copy(os.path.join(root, fixed), new_path)
                    shutil.copy(os.path.join(root, feature_name), new_path)

# format()
# label()
move()