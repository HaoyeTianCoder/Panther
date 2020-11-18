from config_default_old import *
import os
from subprocess import *

class preprocess:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.patch_number = 0

    def get_patch(self, data_path):
        for root, dirs, files in os.walk(data_path):
            if len(files) == 2:
                for file in files:
                    if file.endswith('_s.java'):
                        buggy = os.path.join(root, file)
                    elif file.endswith('_t.java'):
                        patched = os.path.join(root, file)
                    else:
                        self.logger.debug('file name: {}'.format(file))
                patch_name = os.path.join(root, file.split('.')[0][:-2] + '.patch')
                cmd = 'diff -u {} {} > {}'.format(buggy, patched, patch_name)
                try:
                    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                        output, errors = p.communicate(timeout=300)
                        # print(output)
                        if errors:
                            raise CalledProcessError(errors, '-1')
                except Exception as e:
                    self.logger.error('cmd failed to execute. ',exc_info=True)
                    self.logger.info('e: {}'.format(e))
                self.patch_number += 1
                self.logger.info(self.patch_number)

if __name__ == '__main__':
    data_path = Config().ods_data
    p = preprocess()
    p.get_patch(data_path)
