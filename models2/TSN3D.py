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
        inputs = fluid.layers.reshape(x=inputs, shape=[-1, inputs.shape[2], inputs.shape[3], inputs.shape[4], inputs.shape[5]])
        num_seg = inputs.shape[0] // bs

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
                x = fluid.layers.reshape(x=x, shape=[-1, num_seg, x.shape[1], x.shape[2], x.shape[3], x.shape[4]])
                consensusm = consensusmodel.SimpleConsensus(**self.segmental_consensus)
                x = consensusm.net(x)
                x = fluid.layers.squeeze(input=x, axes=[1])

            losses = dict()
            acc = dict()
            if self.cls_head:
                cls_headm = cls_headmodel.ClsHead(**self.cls_head)
                cls_score = cls_headm.net(x)
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
                x = fluid.layers.reshape(x=x, shape=[-1, num_seg, x.shape[1], x.shape[2], x.shape[3], x.shape[4]])
                consensusm = consensusmodel.SimpleConsensus(**self.segmental_consensus)
                x = consensusm.net(x)
                
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



