import paddle.fluid as fluid
import numpy as np

class SimpleSpatialTemporalModule():
    def __init__(self, spatial_type='avg', spatial_size=7, temporal_size=1):
        
        assert spatial_type in ['avg']
        self.spatial_type = spatial_type

        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
        self.temporal_size = temporal_size
        self.pool_size = (self.temporal_size,) + self.spatial_size

    def net(self, input):
        if self.spatial_type == 'avg':
            return fluid.layers.pool3d(input=input, pool_size=self.pool_size, pool_type='avg',
                pool_stride=1, pool_padding=0)

