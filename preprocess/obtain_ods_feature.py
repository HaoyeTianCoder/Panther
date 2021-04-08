import os
import shutil
from subprocess import *

def obtain_ods_features(path_dataset):
    total = 0
    generated = 0
    for root, dirs, files in os.walk(path_dataset):
        for file in files:
            if file.endswith('.patch'):
                total += 1

                name = file.split('.')[0]

                buggy = name + '_s.java'
                fixed = name + '_t.java'

                feature_name = 'features_' + name.split('_')[0] + '.json'
                if os.path.exists(os.path.join(root, feature_name)):
                    generated += 1
                    continue

                frag = root.split('/')
                location = '/'.join(frag[:-2])

                # if name == 'patch1-Closure-43-Developer':
                #     continue

                # cmd = 'java -classpath /Users/haoye.tian/Documents/University/project/coming_tian/' \
                #       'target/coming-0-SNAPSHOT-jar-with-dependencies.jar  fr.inria.coming.main.ComingMain ' \
                #       '-mode features -parameters cross:false -input filespair -location {}:{} -output {}'.format(os.path.join(root,buggy), os.path.join(root,fixed), root)
                cmd = 'java -classpath /Users/haoye.tian/Documents/University/project/coming/' \
                      'target/coming-0-SNAPSHOT-jar-with-dependencies.jar  fr.inria.coming.main.ComingMain ' \
                      '-mode features -parameters cross:false -input files -location {} -output {}'.format(location, root)
                try:
                    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                        output, errors = p.communicate(timeout=300)
                        # print(output)
                        # if errors:
                        #     raise CalledProcessError(errors, '-1')
                        # if output == '' or 'Exception' in output:
                        if not os.path.exists(os.path.join(root, feature_name)) or output == '':
                            print('Name: {}, Error: {}'.format(name, errors))
                            continue
                except Exception as e:
                    print(e)
                    continue
                generated += 1
                print('generated: {}/{}. new: {}'.format(generated, total, name))
    print('total: {}, generated: {}'.format(total, generated))

if __name__ == '__main__':
    path_dataset = '/Users/haoye.tian/Documents/University/data/PatchCollectingV2UniqueToken'
    obtain_ods_features(path_dataset)

