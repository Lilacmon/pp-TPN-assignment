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


if __name__ == '__main__':
    segmental_consensus=dict(
                        consensus_type='avg')
    a = SimpleConsensus(**segmental_consensus)
    data_shape = [1, 2048, 1, 1, 1]
    img = fluid.layers.data(name='images', shape=data_shape, dtype='float32')
    x = a.net(img)
    print(x.shape) #[-1, 1, 2048, 1, 1, 1]
