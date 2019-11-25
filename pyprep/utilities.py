import numpy as np
import logging
import math
from cmath import sqrt
import scipy.interpolate


def union(list1, list2):
    return list(set(list1 + list2))


def set_diff(list1, list2):
    return list(set(list1) - set(list2))

