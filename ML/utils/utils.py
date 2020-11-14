#!usr/bin/python3
import pickle
import datetime
import time
import os
import subprocess

import numpy as np
import pandas as pd
import pandas_profiling
from urllib.parse import unquote

def pickle_dump(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
