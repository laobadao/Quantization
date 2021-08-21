# -*- coding: UTF-8 -*-
from __future__ import division

import os
import sys
import time
import logging


def _get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_path = './log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    tm = time.strftime('%Y%m%d', time.localtime(time.time()))
    log_file = log_path + tm + '.log'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s\
        [line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    logger.addHandler(sh)
    return logger


Log = _get_logger()

LOG_D = Log.debug
LOG_I = Log.info
LOG_W = Log.warning
LOG_E = Log.error
