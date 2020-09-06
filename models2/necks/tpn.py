import paddle.fluid as fluid
import numpy as np
import pdb


class ConvModule():
    def __init__(
            self,
            inplanes,
            planes,
            kernel_size,
            stride,
            padding,
            bias=False,
            groups=1,):
        
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias=bias
        self.groups=groups

    def net(self, x):
        # out = self.bn(self.conv(x))
        out = fluid.layers.conv3d(input=x, num_filters=self.planes, filter_size=self.kernel_size,
                    stride=self.stride, padding=self.padding, groups=self.groups, bias_attr=self.bias)
        out = fluid.layers.batch_norm(input=out, act='relu')

        return out


class AuxHead():
    def __init__(
            self,
            inplanes,
            planes,
            loss_weight=0.5):
        self.inplanes = inplanes
        self.planes = planes
        self.loss_weight = loss_weight

    def net(self, x, target=None):
        if target is None:
            return None
        loss = dict()
        # x = self.convs(x)
        convM = \
            ConvModule(self.inplanes, self.inplanes * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        x = convM.net(x)
        x = fluid.layers.adaptive_pool3d(input=x, pool_size=1, pool_type='avg')
        x = fluid.layers.squeeze(input=x, axes=[2, 3, 4])
        
        x = fluid.layers.dropout(x=x, dropout_prob=0.5)
        x = fluid.layers.fc(input=x, size=self.planes)

        loss['loss_aux'] = self.loss_weight * fluid.layers.softmax_with_cross_entropy(logits=x, label=target)
        # loss['loss_aux'] = self.loss_weight * fluid.layers.cross_entropy(input=x, label=target)
        acc = dict()
        x = fluid.layers.softmax(input=x)
        acc['acc_aux'] = fluid.layers.accuracy(input=x, label=target)

        return loss, acc


class TemporalModulation():
    def __init__(self,
                 inplanes,
                 planes,
                 downsample_scale=8,
                 ):
        self.planes = planes
        self.downsample_scale = downsample_scale

    def net(self, x):
        x = fluid.layers.conv3d(input=x, num_filters=self.planes, filter_size=(3, 1, 1),
            stride=(1, 1, 1), padding=(1, 0, 0), groups=32, bias_attr=False)

        x = fluid.layers.pool3d(input=x, pool_size=(self.downsample_scale, 1, 1), pool_type="max",
                pool_stride=(self.downsample_scale, 1, 1), pool_padding=(0, 0, 0), ceil_mode=True)
        return x


class Upsampling():
    def __init__(self,
                 # scale=(2, 1, 1),
                 scale=[2, 1, 1],
                 ):
        self.scale = scale

    def net(self, x):
        x = fluid.layers.interpolate(input=x, out_shape=[x.shape[2]*self.scale[0],x.shape[3]*self.scale[1],x.shape[4]*self.scale[2]],
                resample='TRILINEAR', data_format='NCDHW')

        return x


class Downampling(fluid.dygraph.Layer):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=(3, 1, 1),
                 stride=(1, 1, 1),
                 padding=(1, 0, 0),
                 bias=False,
                 groups=1,
                 norm=False,
                 activation=False,
                 downsample_position='after',
                 downsample_scale=(1, 2, 2),
                 ):
        
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias

        self.norm = norm
        self.activation = activation
        assert (downsample_position in ['before', 'after'])
        self.downsample_position = downsample_position
        # self.pool = nn.MaxPool3d(downsample_scale, downsample_scale, (0, 0, 0), ceil_mode=True)
        self.downsample_scale = downsample_scale

    def net(self, x):
        if self.downsample_position == 'before':
            x = fluid.layers.pool3d(input=x, pool_size=self.downsample_scale, pool_type="max",
                    pool_stride=self.downsample_scale, pool_padding=(0, 0, 0), ceil_mode=True)
        x = fluid.layers.conv3d(input=x, num_filters=self.planes, filter_size=self.kernel_size,
            stride=self.stride, padding=self.padding, groups=self.groups, bias_attr=self.bias)
        if self.norm:
            x = fluid.layers.batch_norm(input=x)
        
        if self.activation is not None:
            x = fluid.layers.relu(x)
        if self.downsample_position == 'after':
            x = fluid.layers.pool3d(input=x, pool_size=self.downsample_scale, pool_type="max",
                    pool_stride=self.downsample_scale, pool_padding=(0, 0, 0), ceil_mode=True)

        return x


class LevelFusion():
    def __init__(self,
                 in_channels=[1024, 1024],
                 mid_channels=[1024, 1024],
                 out_channels=2048,
                 ds_scales=[(1, 1, 1), (1, 1, 1)]):

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.ds_scales = ds_scales
        self.out_channels = out_channels

    def net(self, inputs):
        # out = [self.ops[i](feature) for i, feature in enumerate(inputs)]
        out = []
        for i, feature in enumerate(inputs):
            downm = Downampling(self.in_channels[i], self.mid_channels[i], kernel_size=(1, 1, 1), stride=(1, 1, 1),
                             padding=(0, 0, 0), bias=False, groups=32, norm=True, activation=True,
                             downsample_position='before', downsample_scale=self.ds_scales[i])
            out.append(downm.net(feature))

        out = fluid.layers.concat(input=out, axis=1)
        out = fluid.layers.conv3d(input=out, num_filters=self.out_channels,filter_size=1, stride=1, padding=0, bias_attr=False)
        out = fluid.layers.batch_norm(input=out, act='relu')
        return out


class SpatialModulation(fluid.dygraph.Layer):
    def __init__(
            self,
            inplanes=[1024, 2048],
            planes=2048):
        self.inplanes=inplanes
        self.planes = planes   

    def net(self, inputs):
        out = []
        # for i, feature in enumerate(inputs):
        #     if isinstance(self.spatial_modulation[i], list):
        #         out_ = inputs[i]
        #         for III, op in enumerate(self.spatial_modulation[i]):
        #             out_ = op(out_)
        #         out.append(out_)
        #     else:
        #         out.append(self.spatial_modulation[i](inputs[i]))
        for i, dim in enumerate(self.inplanes):
            ds_factor = self.planes // dim
            ds_num = int(np.log2(ds_factor))
            if ds_num < 1:
                # identity = Identity()
                out.append(inputs[i])

            else:
                for dsi in range(ds_num):
                    in_factor = 2 ** dsi
                    out_factor = 2 ** (dsi + 1)
                    convM = ConvModule(dim * in_factor, dim * out_factor, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                         padding=(0, 1, 1), bias=False)
                    out_ = convM.net(inputs[i])
                    out.append(out_)    

        return out


class TPN():
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 spatial_modulation_config=None,
                 temporal_modulation_config=None,
                 upsampling_config=None,
                 downsampling_config=None,
                 level_fusion_config=None,
                 aux_head_config=None,
                 ):
        assert isinstance(in_channels, list)
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        aux_head_config['inplanes'] = self.in_channels[-2]
        self.aux_head_config = aux_head_config
        self.spatial_modulation_config = spatial_modulation_config
        self.temporal_modulation_config = temporal_modulation_config
        self.upsampling_config = upsampling_config
        self.downsampling_config = downsampling_config
        self.level_fusion_config = level_fusion_config

    def net(self, inputs, target=None, is_training=None):
        loss = None
        # Auxiliary loss
        if self.aux_head_config is not None and is_training:
            auxHeadm = AuxHead(**self.aux_head_config)
            loss, acc = auxHeadm.net(inputs[-2], target)

        # Spatial Modulation
        spatialm = SpatialModulation(**self.spatial_modulation_config)
        outs = spatialm.net(inputs)

        # Temporal Modulation
        # outs = [temporal_modulation(outs[i]) for i, temporal_modulation in enumerate(self.temporal_modulation_ops)]
        outs_t = []
        for i in range(0, self.num_ins, 1):
            inplanes = self.in_channels[-1]
            planes = self.out_channels
            if self.temporal_modulation_config is not None:
                self.temporal_modulation_config['param']['downsample_scale'] = self.temporal_modulation_config['scales'][i]
                self.temporal_modulation_config['param']['inplanes'] = inplanes
                self.temporal_modulation_config['param']['planes'] = planes
                temporalm = TemporalModulation(**self.temporal_modulation_config['param'])
                outs_t.append(temporalm.net(outs[i]))
        
        outs = outs_t
        temporal_modulation_outs = outs
        
        # Build top-down flow - upsampling operation
        if self.upsampling_config is not None:
            for i in range(self.num_ins - 1, 0, -1):
                # outs[i - 1] = outs[i - 1] + self.upsampling_ops[i - 1](outs[i])
                upsamplingm = Upsampling(self.upsampling_config.get('scale'))
                outs[i - 1] = outs[i - 1] + upsamplingm.net(outs[i])

        # Get top-down outs
        levelFusionm = LevelFusion(**self.level_fusion_config)
        topdownouts = levelFusionm.net(outs)
        outs = temporal_modulation_outs

        # Build bottom-up flow - downsampling operation
        if self.downsampling_config is not None:
            for i in range(0, self.num_ins - 1, 1):
                planes = self.out_channels
                # outs[i + 1] = outs[i + 1] + self.downsampling_ops[i](outs[i])
                self.downsampling_config['param']['inplanes'] = planes
                self.downsampling_config['param']['planes'] = planes
                self.downsampling_config['param']['downsample_scale'] = self.downsampling_config['scales']
                downsamplingm = Downampling(**self.downsampling_config['param'])
                outs[i + 1] = outs[i + 1] + downsamplingm.net(outs[i])

        # Get bottom-up outs
        levelfm = LevelFusion(**self.level_fusion_config)
        outs = levelfm.net(outs)

        # fuse two pyramid outs
        # outs = self.pyramid_fusion_op(fluid.layers.concat(input=[topdownouts, outs], axis=1))
        outs = fluid.layers.conv3d(input=fluid.layers.concat(input=[topdownouts, outs], axis=1),
                num_filters=2048, filter_size=1, stride=1, padding=0, bias_attr=False)
        outs = fluid.layers.batch_norm(input=outs, act='relu')
        
        if is_training:
            return outs, loss, acc
        else:
            return outs





    
