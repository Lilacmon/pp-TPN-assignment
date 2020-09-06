import paddle.fluid as fluid
import numpy as np
import pdb
import models2.backbones.resnet_slow as backmodel
import models2.necks.tpn as neckmodel
import models2.spatial_temporal_modules.simple_spatial_temporal_modules as stmodel
import models2.segmental_consensuses.simple_consensus as consensusmodel
import models2.cls_heads.cls_head as cls_headmodel


class TSN3D():
    def __init__(self,
                 backbone,
                 necks=None,
                 spatial_temporal_module=None,
                 segmental_consensus=None,
                 fcn_testing=False,
                 flip=False,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None):

        self.backbone = backbone
        self.necks = necks
        self.spatial_temporal_module = spatial_temporal_module
        self.segmental_consensus = segmental_consensus
        self.fcn_testing = fcn_testing
        self.flip = flip
        self.cls_head = cls_head
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    def net(self, inputs, label=None, is_training_val=None):
        bs = inputs.shape[0]
        # inputs = inputs.reshape((-1,) + inputs.shape[2:])
        inputs = fluid.layers.reshape(x=inputs, shape=[-1, inputs.shape[2], inputs.shape[3], inputs.shape[4], inputs.shape[5]])
        num_seg = inputs.shape[0] // bs

        # num_seg = inputs.shape[1]
        # x = self.backbone(inputs)
        backm = backmodel.ResNet_SlowFast(**self.backbone)
        x = backm.net(inputs)

        if is_training_val :
            if self.necks is not None:
                neckm = neckmodel.TPN(**self.necks)
                x, aux_losses, aux_acc = neckm.net(x, label, is_training_val)

            if self.spatial_temporal_module:
                stm = stmodel.SimpleSpatialTemporalModule(**self.spatial_temporal_module)
                x = stm.net(x)

            if self.segmental_consensus:
                # x = x.reshape((-1, num_seg) + x.shape[1:])
                x = fluid.layers.reshape(x=x, shape=[-1, num_seg, x.shape[1], x.shape[2], x.shape[3], x.shape[4]])
                consensusm = consensusmodel.SimpleConsensus(**self.segmental_consensus)
                x = consensusm.net(x)
                # x = x.squeeze(1)
                x = fluid.layers.squeeze(input=x, axes=[1])

            losses = dict()
            acc = dict()
            if self.cls_head:
                cls_headm = cls_headmodel.ClsHead(**self.cls_head)
                cls_score = cls_headm.net(x)
                # gt_label = gt_label.squeeze()
                # label = fluid.layers.squeeze(input=label, axes = [])
                loss_cls, acc_cls = cls_headm.loss(cls_score, label)
                losses.update(loss_cls)
                acc.update(acc_cls)

            if self.necks is not None:
                if aux_losses is not None:
                    losses.update(aux_losses)
                    acc.update(aux_acc)

            return losses, acc
        else:
            if self.necks is not None:
                neckm = neckmodel.TPN(**self.necks)
                x = neckm.net(x, label, is_training_val)

            if self.spatial_temporal_module:
                stm = stmodel.SimpleSpatialTemporalModule(**self.spatial_temporal_module)
                x = stm.net(x)
            
            if self.segmental_consensus:
                # x = x.reshape((-1, num_seg) + x.shape[1:])
                x = fluid.layers.reshape(x=x, shape=[-1, num_seg, x.shape[1], x.shape[2], x.shape[3], x.shape[4]])
                consensusm = consensusmodel.SimpleConsensus(**self.segmental_consensus)
                x = consensusm.net(x)
                # x = x.squeeze(1)
                x = fluid.layers.squeeze(input=x, axes=[1])

            if self.cls_head:
                cls_headm = cls_headmodel.ClsHead(**self.cls_head)
                x = cls_headm.net(x)

            x = fluid.layers.softmax(input=x)
            if label is not None:
                acc_val = fluid.layers.accuracy(input=x, label=label)
                return x, acc_val
            else:
                return x

if __name__ == '__main__':
    network = TSN3D(backbone=dict(
                        # pretrained='modelzoo://resnet50',
                        depth=50,
                        num_stages=4,
                        out_indices=[2, 3],
                        frozen_stages=-1,
                        inflate_freq=(0, 0, 1, 1),
                        inflate_style='3x1x1',
                        conv1_kernel_t=1,
                        conv1_stride_t=1,
                        pool1_kernel_t=1,
                        pool1_stride_t=1,
                        with_cp=True,
                        bn_eval=False,
                        partial_bn=False,
                        style='pytorch'),
                    necks=dict(
                        in_channels=[1024, 2048],
                        out_channels=1024,
                        spatial_modulation_config=dict(
                            inplanes=[1024, 2048],
                            planes=2048,
                        ),
                        temporal_modulation_config=dict(
                            scales=(32, 32),
                            param=dict(
                                inplanes=-1,
                                planes=-1,
                                downsample_scale=-1,
                            )),
                        upsampling_config=dict(
                            scale=(1, 1, 1),
                        ),
                        downsampling_config=dict(
                            scales=(1, 1, 1),
                            param=dict(
                                inplanes=-1,
                                planes=-1,
                                downsample_scale=-1,
                            )),
                        level_fusion_config=dict(
                            in_channels=[1024, 1024],
                            mid_channels=[1024, 1024],
                            out_channels=2048,
                            ds_scales=[(1, 1, 1), (1, 1, 1)],
                        ),
                        aux_head_config=dict(
                            inplanes=-1,
                            planes=400,
                            loss_weight=0.5
                        ),
                    ),
                    spatial_temporal_module=dict(
                        spatial_type='avg',
                        temporal_size=1,
                        spatial_size=7),
                    segmental_consensus=dict(
                        consensus_type='avg'),
                    cls_head=dict(
                        with_avg_pool=False,
                        temporal_feature_size=1,
                        spatial_feature_size=1,
                        dropout_ratio=0.5,
                        in_channels=2048,
                        num_classes=400))

    # img = np.zeros([2, 1, 3, 32, 224, 224]).astype('float32')
    # y_data = np.array([[1],[2]]).astype('int64')
    # img = fluid.dygraph.to_variable(img)
    # label = fluid.dygraph.to_variable(y_data)
    data_shape = [1, 3, 32, 224, 224]
    img = fluid.layers.data(name='images', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    outs = network.net(img, label, is_training_val=True)
    pdb.set_trace()
    print(outs) 

    # backmodel
    # outs[0].shape   [2, 1024, 32, 14, 14]
    # outs[1].shape   [2, 2048, 32, 7, 7]

    # necks
    # (<paddle.fluid.core_avx.VarBase object at 0x7f9ec39d7970>, {'loss_aux': <paddle.fluid.core_avx.VarBase object at 0x7f9ec3a248b0>})
    # outs[0].shape   [2, 2048, 1, 7, 7]
    # outs[1].get('loss_aux').shape)  [2, 1]

    # spatial_temporal_module
    # outs.shape [2, 2048, 1, 1, 1]

    # segmental_consensus
    # [2, 2048, 1, 1, 1]   [2, num_seg=1, 2048, 1, 1, 1]
    # [2, 2048, 1, 1, 1]


