import paddle.fluid as fluid
import numpy as np
import pdb

def block(x,
        inplanes,
        planes,
        spatial_stride=1,
        temporal_stride=1,
        dilation=1,
        is_downsample=False,
        # style='pytorch',
        # style='paddle',
        if_inflate=True,
        inflate_style='3x1x1',
        if_nonlocal=True,
        nonlocal_cfg=None,
        with_cp=False):

    identity = x
    # out = self.conv1(x)
    # out = self.bn1(out)
    conv1_stride = 1
    conv2_stride = spatial_stride
    conv1_stride_t = 1
    conv2_stride_t = temporal_stride
    if if_inflate:
        if inflate_style == '3x1x1':
            out = fluid.layers.conv3d(input=x, num_filters=planes, filter_size=(3, 1, 1),
                    stride=(conv1_stride_t, conv1_stride, conv1_stride), padding=(1, 0, 0), bias_attr=False)
            out = fluid.layers.batch_norm(input=out, act='relu')
            out = fluid.layers.conv3d(input=out, num_filters=planes, filter_size=(1, 3, 3),
                stride=(conv2_stride_t, conv2_stride, conv2_stride), padding=(0, dilation, dilation),
                dilation=(1, dilation, dilation), bias_attr=False)
            out = fluid.layers.batch_norm(input=out, act='relu')
        else:
            out = fluid.layers.conv3d(input=x, num_filters=planes, filter_size=1,
                    stride=(conv1_stride_t, conv1_stride, conv1_stride), bias_attr=False)
            out = fluid.layers.batch_norm(input=out, act='relu')
            out = fluid.layers.conv3d(input=out, num_filters=planes, filter_size=3,
                stride=(conv2_stride_t, conv2_stride, conv2_stride), padding=(1, dilation, dilation),
                dilation=(1, dilation, dilation), bias_attr=False)
            out = fluid.layers.batch_norm(input=out, act='relu')
    else:
        out = fluid.layers.conv3d(input=x, num_filters=planes, filter_size=1,
                    stride=(1, conv1_stride, conv1_stride), bias_attr=False)
        out = fluid.layers.batch_norm(input=out, act='relu')
        out = fluid.layers.conv3d(input=out, num_filters=planes, filter_size=(1, 3, 3),
            stride=(1, conv2_stride, conv2_stride), padding=(0, dilation, dilation),
            dilation=(1, dilation, dilation), bias_attr=False)
        out = fluid.layers.batch_norm(input=out, act='relu')

    out = fluid.layers.conv3d(input=out, num_filters=planes * 4, filter_size=1, bias_attr=False)
    out = fluid.layers.batch_norm(input=out)
    if is_downsample:
        identity = fluid.layers.conv3d(input=identity, num_filters=planes * 4, filter_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride), bias_attr=False)
        identity = fluid.layers.batch_norm(input=identity)

    out += identity
    out = fluid.layers.relu(out)

    return out


def make_res_layer(x,
                #    block,
                   inplanes,
                   planes,
                   blocks,
                   spatial_stride=1,
                   temporal_stride=1,
                   dilation=1,
                   # style='pytorch',
                #    style='paddle',
                   inflate_freq=1,
                   inflate_style='3x1x1',
                   nonlocal_freq=1,
                   nonlocal_cfg=None,
                   with_cp=False):
    inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * blocks
    nonlocal_freq = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * blocks
    assert len(inflate_freq) == blocks
    assert len(nonlocal_freq) == blocks
    is_downsample = False
    if spatial_stride != 1 or inplanes != planes * 4:  # block.expansion:
        is_downsample = True
        # x = fluid.layers.conv3d(input=x, num_filters=planes * 4, filter_size=1,
        #         stride=(temporal_stride, spatial_stride, spatial_stride), bias_attr=False)
        # x = fluid.layers.batch_norm(input=x)
    
    x = block(
        x,
        inplanes,
        planes,
        spatial_stride,
        temporal_stride,
        dilation,
        is_downsample,
        # style=style,
        if_inflate=(inflate_freq[0] == 1),
        inflate_style=inflate_style,
        if_nonlocal=(nonlocal_freq[0] == 1),
        nonlocal_cfg=nonlocal_cfg,
        with_cp=with_cp)
    inplanes = planes * 4  # block.expansion
    for i in range(1, blocks):
        x = block(x,
                inplanes,
                planes,
                1, 1,
                dilation,
                # style=style,
                if_inflate=(inflate_freq[i] == 1),
                inflate_style=inflate_style,
                if_nonlocal=(nonlocal_freq[i] == 1),
                nonlocal_cfg=nonlocal_cfg,
                with_cp=with_cp)

    return x


class ResNet_SlowFast():
    def __init__(self,
                 depth,
                 pretrained=None,
                 pretrained2d=True,
                 num_stages=4,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 conv1_kernel_t=5,
                 conv1_stride_t=2,
                 pool1_kernel_t=1,
                 pool1_stride_t=2,
                 style='pytorch',
                 frozen_stages=-1,
                 inflate_freq=(1, 1, 1, 1),
                 inflate_stride=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 nonlocal_stages=(-1,),
                 nonlocal_freq=(0, 1, 1, 0),
                 nonlocal_cfg=None,
                 bn_eval=False,
                 bn_frozen=False,
                 partial_bn=False,
                 with_cp=False):

        self.arch_settings = {
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3)
        }

        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.inflate_freqs = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * num_stages
        self.inflate_style = inflate_style
        self.nonlocal_stages = nonlocal_stages
        self.nonlocal_freqs = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * num_stages
        self.nonlocal_cfg = nonlocal_cfg
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp

        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self.pool1_kernel_t = pool1_kernel_t
        self.pool1_stride_t = pool1_stride_t

        self.conv1_kernel_t = conv1_kernel_t
        self.conv1_stride_t = conv1_stride_t


    def net(self, x):
        x = fluid.layers.conv3d(input=x, num_filters=64, filter_size=(self.conv1_kernel_t, 7, 7),
            stride=(self.conv1_stride_t, 2, 2), padding=((self.conv1_kernel_t - 1) // 2, 3, 3), bias_attr=False)
        x = fluid.layers.batch_norm(input=x, act='relu')
        x = fluid.layers.pool3d(input=x, pool_size=(self.pool1_kernel_t, 3, 3), pool_type="max",
            pool_stride=(self.pool1_stride_t, 2, 2), pool_padding=(self.pool1_kernel_t // 2, 1, 1))

        outs = []
        # for i, layer_name in enumerate(self.res_layers):
        # i = 0
        # for res_layer in self.res_layers:
        #     # res_layer = getattr(self, layer_name)
        #     x = res_layer(x)
        #     if i in self.out_indices:
        #         outs.append(x)
        #     i += 1
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = self.spatial_strides[i]
            temporal_stride = self.temporal_strides[i]
            dilation = self.dilations[i]
            planes = 64 * 2 ** i
            x = make_res_layer(
                x,
                # self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                # style=self.style,
                inflate_freq=self.inflate_freqs[i],
                inflate_style=self.inflate_style,
                nonlocal_freq=self.nonlocal_freqs[i],
                nonlocal_cfg=self.nonlocal_cfg if i in self.nonlocal_stages else None,
                with_cp=self.with_cp)
            self.inplanes = planes * 4 # self.block.expansion
            # layer_name = 'layer{}'.format(i + 1)
            # self.res_layers.append(self.add_sublayer(layer_name, res_layer))
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

if __name__ == '__main__':
    
    network = ResNet_SlowFast(
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
                style='paddle')
    # img = np.zeros([1, 3, 32, 224, 224]).astype('float32')
    # img = fluid.dygraph.to_variable(img)
    data_shape = [3, 32, 224, 224]
    img = fluid.layers.data(name='images', shape=data_shape, dtype='float32')
    outs = network.net(img)
    print(outs)
    print(outs[0].shape, outs[1].shape)
