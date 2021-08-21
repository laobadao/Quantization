# -*- coding: utf-8 -*-
import re
import math
import numpy as np

from typing import Any, List, Dict, Tuple
from utils.log import LOG_E, LOG_I


def quantize_int(data: np.ndarray, bits: int) -> Tuple:
    int_scale_ = float((1 << (bits - 1)))
    max_int_ = (1 << (bits - 1)) - 1

    r_d = np.abs(np.reshape(data, (-1, data.shape[-1])))
    data_max = np.max(r_d, axis=0) * int_scale_ / max_int_
    data_max[data_max < 1e-3] = 1e-3

    w_revert = data.copy() * int_scale_ / data_max
    w_revert = np.round(w_revert, 0)

    # [-127, 127]
    w_revert = np.maximum(w_revert, -max_int_)
    w_revert = np.minimum(w_revert, max_int_)

    return w_revert / int_scale_, data_max


def quantize_uint(data: np.ndarray, bits: int) -> Tuple:
    if bits < 0:
        LOG_E('param[bits:{}] Err!!!'.format(bits))
        raise Exception('PARAM ERR')

    max_f = float((1 << (bits)))

    d = np.reshape(data, (-1, data.shape[-1]))
    l_max = np.max(d, axis=0)
    l_min = np.min(d, axis=0)
    l_step = (l_max - l_min) / max_f

    w_uint = np.round((data - l_min) / l_step, 0) * l_step + l_min
    l_max = np.ones(l_max.shape)
    return w_uint, l_max


def quantize_sfp(data: np.ndarray, int_bits: int, decimal_bits: int) -> Tuple:
    if int_bits < 0 or decimal_bits < 0:
        LOG_E('param[int_bits:{}, decimal_bits:{}] Err!!!'.format(int_bits,
                                                                  decimal_bits))
        raise Exception('PARAM ERR')

    all_bits = int_bits + decimal_bits
    max_int_v = (1 << all_bits) - 1
    print(max_int_v)
    min_int_v = (1 << all_bits) * -1
    print(min_int_v)
    array_data = data.copy() * (1 << decimal_bits)
    print(array_data)
    array_data = np.round(array_data, 0)
    array_data[array_data > max_int_v] = max_int_v
    array_data[array_data < min_int_v] = min_int_v
    array_data = array_data / (1 << decimal_bits)
    return array_data, 1


def quantify_data(compress_mode: str, data: np.ndarray) -> Tuple:
    max_data = np.max(np.abs(data))

    m = re.match(r'SFP(\d+)', compress_mode)
    if m:
        bit = int(m.group(1))
        if bit not in [8, 16]:
            raise Exception('SFP_MODE({}) ERR'.format(compress_mode))

        if max_data == 0:
            return data, 1, 0, bit - 1

        q = int(math.ceil(np.log2(max_data)))
        q = min(max(0, q), bit - 1)
        (array_data, max_) = QuantSFP(data, q, bit - 1 - q)
        return array_data, max_, q, bit - 1 - q

    m = re.match(r'INT(\d+)', compress_mode)
    if m:
        bit = int(m.group(1))
        if bit not in [8]:
            raise Exception('INT_MODE({}) ERR'.format(compress_mode))

        if max_data == 0:
            return data, 1, 0, bit - 1

        (array_data, max_) = QuantInt(data, bit)
        return array_data, max_, 0, bit - 1

    m = re.match(r'UINT(\d+)', compress_mode)
    if m:
        bit = int(m.group(1))
        if bit not in [8]:
            raise Exception('UINT_MODE({}) ERR'.format(compress_mode))

        if max_data == 0:
            return (data, 1, 0, bit)

        (array_data, max_) = QuantUInt(data, bit)
        return array_data, max_, 0, bit

    LOG_E('param[compress_mode:{}] Err!!!'.format(compress_mode))
    raise Exception('QUANT_MODE ERR')
    return


def numpy2tdnn(inputdata: np.ndarray, old_frame_count: int, batch_frame_count: int, stride=1):
    inputw = 0
    if len(inputdata.shape) == 4:
        inputw = inputdata.shape[2]
    if len(inputdata.shape) == 3:
        inputw = inputdata.shape[1]
    if inputw == batch_frame_count:
        return inputdata
    tmp = []
    dim = inputdata.shape
    inputdata = inputdata.transpose(0, 2, 1, 3)
    for i in range(old_frame_count):
        tmp.append(inputdata[0, i, :, :])

    for i in range(1, dim[0]):
        tmp.append(inputdata[i, -stride, :, :])
    ret = []
    leftframe = old_frame_count - batch_frame_count
    begin_pos = 0
    for batch in range(dim[0] + leftframe):
        data = tmp[begin_pos:begin_pos + batch_frame_count]
        ret.append(data)
        begin_pos = begin_pos + stride

    ret = np.array(ret)
    ret = ret.transpose(0, 2, 1, 3)
    print("ret", ret.shape)
    return ret
