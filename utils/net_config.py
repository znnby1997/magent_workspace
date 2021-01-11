import sys
sys.path.append('..')

from net.base import BasicNet
from net.alw_att_net import AlwAttNet
from net.dot_scale_att_net import DotScaleAttNet
from net.ian import IAN


net_cfg = {
    'none': BasicNet,
    'alw': AlwAttNet,
    'dsan': DotScaleAttNet,
    'ian': IAN
}