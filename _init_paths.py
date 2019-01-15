import os.path as path
import sys


def add_path(_path):
    if path not in sys.path:
        sys.path.insert(0, _path)


this_dir = path.dirname(__file__)

print(this_dir)
