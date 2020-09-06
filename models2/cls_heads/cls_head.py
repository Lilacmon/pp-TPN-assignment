import paddle.fluid as fluid
import numpy as np
import pdb


class ClsHead():
    def __init__(self,
                 with_avg_pool=True,
                 temporal_feature_size=1,
                 spatial_feature_size=7,
                 dropout_ratio=0.8,
                 in_channels=2048,
                 num_classes=101,
                 fcn_testing=False,
                 init_std=0.01):

        self.with_avg_pool = with_avg_pool
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.temporal_feature_size = temporal_feature_size
        self.spatial_feature_size = spatial_feature_size
        self.init_std = init_std
        self.fcn_testing = fcn_testing
        
        if self.fcn_testing:
            self.new_cls = None
            self.in_channels = in_channels
            self.num_classes = num_classes
        
        self.num_classes = num_classes

    def net(self, x):
        assert x.shape[1] == self.in_channels
        assert x.shape[2] == self.temporal_feature_size
        assert x.shape[3] == self.spatial_feature_size
        assert x.shape[4] == self.spatial_feature_size
        if self.with_avg_pool:
            x = fluid.layers.pool3d(input=x,
                    pool_size=(self.temporal_feature_size, self.spatial_feature_size, self.spatial_feature_size),
                    pool_type='avg', pool_stride=(1, 1, 1), pool_padding=(0, 0, 0))

        if self.dropout_ratio != 0:
            x = fluid.layers.dropout(x=x, dropout_prob=self.dropout_ratio)
        
        # x = fluid.layers.reshape(x=x, shape=[x.shape[0], -1])
        x = fluid.layers.reshape(x=x, shape=[x.shape[0], x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]])
        cls_score = fluid.layers.fc(input=x, size=self.num_classes)
        
        return cls_score   

    def loss(self,
             cls_score,
             labels):
        losses = dict()
        losses['loss_cls'] = fluid.layers.softmax_with_cross_entropy(logits=cls_score, label=labels)
        
        acc = dict()
        cls_score = fluid.layers.softmax(input=cls_score)
        acc['acc_cls'] = fluid.layers.accuracy(input=cls_score, label=labels)

        return losses, acc


if __name__ == '__main__': 
    cls_head=dict(
            with_avg_pool=False,
            temporal_feature_size=1,
            spatial_feature_size=1,
            dropout_ratio=0.5,
            in_channels=2048,
            num_classes=400)
    a = ClsHead(**cls_head)
    data_shape = [2048, 1, 1, 1]
    img = fluid.layers.data(name='images', shape=data_shape, dtype='float32')
    x = a.net(img) 
    print(x.shape) # [2, 400]
