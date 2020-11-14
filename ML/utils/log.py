import sys
import time
import os
import os.path as osp


class Logger(object):
    def __init__(self, output_dir, file_name=None):
        self.log_dir = output_dir
        self.terminal = sys.stdout
        ensure_dir(self.log_dir)
        if file_name is None:
            file_name = str(int(time.time()))

        log_file = osp.join(self.log_dir, file_name + ".log")

        self.log = open(log_file, "a")

    def write(self, message):
        if message == '\n':
            return
        cur_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        message = f'{cur_time}: {message}\n'
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    pass
