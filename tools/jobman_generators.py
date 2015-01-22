"""
Module for general generators for jobman experiments.
"""

import collections
from itertools import product
from math import exp
from math import log

def hidden_generator(param_name, num, scale=100):
    assert num > 0
    assert isinstance(num, int)
    assert isinstance(scale, int)
    for hid in xrange(scale, scale * (num + 1), scale):
        yield (param_name, hid)

def float_generator(param_name, num, start, finish, log_scale=False):
    assert isinstance(num, int)
    for n in xrange(num):
        if log_scale:
            yield (param_name, (abs(start) // start) * exp(log(abs(start)) + float(n) / float(num - 1) * (log(abs(finish)) - log(abs(start)))))
        else:
            yield (param_name, (start + float(n) / float(num - 1) * (finish - start)))

def layer_depth_generator(param_name, num, depths):
    assert isinstance(depths, (int, collections.Iterable))
    assert isinstance(num, (int, collections.Iterable))
    if isinstance(depths, int):
        depth_iterator = xrange(depths, depths + 1)
    if isinstance(num, int):
        iterator = xrange(num, num + 1)
    else:
        iterator = num

    for num in iterator:
        assert num > 1
        for outer in depth_iterator:
            for inner in depth_iterator:
                yield (param_name, [outer] + [inner] * (num - 2) + [outer])

def nested_generator(*args):
    for x in product(*args):
        yield x

def list_generator(param_name, l):
    for i in l:
        yield (param_name, i)
