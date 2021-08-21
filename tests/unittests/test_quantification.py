#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

@Author     : Junjun Zhao
@Email      : jjzhao@intenginetech.com
@File       : test_quantification.py
@IDE        : PyCharm
@Modify Time: 2021/8/21 17:01

"""

import numpy as np


def test_symmetrical_uniform_distribution():
    from quantize_core import  quantize_sfp

    fp32_data = np.array([10, -10, 0.11, 0.21, 0.15, 0.05, -0.14, -0.22, -0.08, -0.35])
    max_value = np.max(fp32_data)
    min_value = np.min(fp32_data)
    equidistant_distance = (max_value - min_value)/(2**8-1)
    print("*** equidistant_distance:", equidistant_distance)
    int8_data = (fp32_data/max_value)*127
    print(int8_data)
    print(np.round(int8_data))

    # fixed_int8 = quantize_sfp(fp32_data, 7, 0)
    # print(fixed_int8)



test_symmetrical_uniform_distribution()
