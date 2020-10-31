import os
import shutil

data_path = '/Users/haoye.tian/Documents/University/data/Exp-2-data'
data_path = '/Users/haoye.tian/Downloads/ODS/data'

def deduplicate(data_path):
    unique = set()
    total = 0
    for root, dirs, files in os.walk(data_path):
        for file in files:
            # if not file.endswith('.txt'):
            if not file.endswith('.patch'):
                continue
            total += 1
            content = ''
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                try:
                    for line in f:
                        if line.startswith('---') or line.startswith('+++'):
                            continue
                        # elif line.startswith('@@'):
                        #     continue
                        else:
                            content += line
                except Exception as e:
                    print(e)
                    continue
            if content not in unique:
                # file_path_new = file_path.replace('Exp-2-data','Exp-2-data-deduplicate')
                file_path_new = file_path.replace('data','data-deduplicate')
                dir_new = '/'.join(file_path_new.split('/')[:-1])
                if not os.path.exists(dir_new):
                    os.makedirs(dir_new)
                shutil.copyfile(file_path, file_path_new)

                # add new patch
                unique.add(content)

    print('original number: {}, deduplicate number: {}'.format(total, len(unique)))




if __name__ == '__main__':
    deduplicate(data_path)