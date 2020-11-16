import os
import shutil
from subprocess import *

path = '/Users/haoye.tian/Documents/University/data/PatchCollecting/'
path2 = '/Users/haoye.tian/Documents/University/data/PatchCollectingV2/'

def prepare_legal_file(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.')[0]

                new_path = root.replace('PatchCollecting','PatchCollectingV2')
                if not os.path.exists(new_path):
                    os.makedirs(new_path)

                old_patch = new_patch = file

                old_buggy = name + '.buggy'
                old_fixed = name + '.fixed'

                new_buggy = name + '-s.java'
                new_fixed = name + '-t.java'

                shutil.copy(os.path.join(root, old_patch), os.path.join(new_path, new_patch))
                shutil.copy(os.path.join(root, old_buggy), os.path.join(new_path, new_buggy))
                shutil.copy(os.path.join(root, old_fixed), os.path.join(new_path, new_fixed))

def obtain_ods_features(path):
    total = 2051
    generated = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.')[0]

                buggy = name + '-s.java'
                fixed = name + '-t.java'

                feature_name = 'features_' + buggy + '->' + fixed + '.json'
                if os.path.exists(os.path.join(root, feature_name)):
                    generated += 1
                    continue

                cmd = 'java -classpath /Users/haoye.tian/Documents/University/project/coming_tian/' \
                      'target/coming-0-SNAPSHOT-jar-with-dependencies.jar  fr.inria.coming.main.ComingMain ' \
                      '-mode features -input filespair -location {}:{} -output {}'.format(os.path.join(root,buggy), os.path.join(root,fixed), root)
                try:
                    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                        output, errors = p.communicate(timeout=300)
                        # print(output)
                        # if errors:
                        #     raise CalledProcessError(errors, '-1')
                        if output == '' or 'error' in errors or 'ERROR' in errors:
                            print('error')
                            continue
                except Exception as e:
                    print(e)
                    continue
                generated += 1
                print('generated: {}, new:{}'.format(generated, name))
    print('total: {}, generated: {}'.format(total, generated))


# prepare_legal_file(path)
obtain_ods_features(path2)

