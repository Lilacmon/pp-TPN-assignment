import paddle.fluid as fluid
import numpy as np
import pdb


class _SimpleConsensus():
    def __init__(self,
                 consensus_type='avg',
                 dim=1):

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def net(self, x):
        self.shape = x.shape
        if self.consensus_type == 'avg':
            output = fluid.layers.reduce_mean(x, dim=self.dim, keep_dim=True)
        else:
            output = None
        return output


class SimpleConsensus():
    def __init__(self, consensus_type, dim=1):

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim

    def net(self, input):
        # return _SimpleConsensus(self.consensus_type, self.dim)(input)
        consensus = _SimpleConsensus(self.consensus_type, self.dim)
        return consensus.net(input)

