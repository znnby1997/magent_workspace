import sys
sys.path.append('..')

from net.base import BasicNet
from net.alw_att_net import AlwAttNet
# from net.dot_scale_att_net import DotScaleAttNet
# from net.ian import IAN
from net.odfsn_base import ODFSNBase
from net.odfsn_self import ODFSNSelf


net_cfg = {
    'none': BasicNet,
    'oan': AlwAttNet, # observation attention network
    # 'dsan': DotScaleAttNet,
    # 'ian': IAN
    'odfsn_base': ODFSNBase,
    'odfsn_self': ODFSNSelf
}