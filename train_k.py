import paddle as paddle
import paddle.fluid as fluid
import numpy as np
from PIL import Image
import os
import zipfile
import linecache
import argparse
import ast
from config import parse_config, merge_configs, print_configs
from reader import KineticsReader
import models2.TSN3D as TSN



# 数据集文件目录
datasets_prefix = '/root/paddlejob/workspace/train_data/datasets/'
# 数据集文件具体路径请在编辑项目状态下,通过左侧导航栏「数据集」中文件路径拷贝按钮获取
train_datasets =  datasets_prefix + 'data49371/k400.zip'  # 'data19627/hmdb51_org.zip' 
# 输出文件目录. 任务完成后平台会自动把该目录所有文件压缩为tar.gz包，用户可以通过「下载输出」可以将输出信息下载到本地.
output_dir = "/root/paddlejob/workspace/output"

# 新建解压文件夹
dest_dir = "/root/paddlejob/workspace/datasets/"
os.system("mkdir " + dest_dir)

# myzip = 'k400test.zip'
# 解压数据集
if zipfile.is_zipfile(train_datasets):  # 检查是否为zip文件
    with zipfile.ZipFile(train_datasets, 'r') as zipf:
        zipf.extractall(dest_dir)
    print('unzip success.')
    
rpaths = os.listdir(dest_dir)
for i in rpaths:
    print(dest_dir, i)

os.system('cp /root/paddlejob/workspace/datasets/trainlist.txt /root/paddlejob/workspace/output/trainlist.txt')
os.system('cp /root/paddlejob/workspace/datasets/vallist.txt /root/paddlejob/workspace/output/vallist.txt')

print(linecache.getline('/root/paddlejob/workspace/datasets/trainlist.txt', 1))
print(linecache.getline('/root/paddlejob/workspace/datasets/vallist.txt', 1))

def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='tsn',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/p2_tpn.txt',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=1,  # 100,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/root/paddlejob/workspace/output',
        help='directory name to save train snapshoot')
    args = parser.parse_args()
    return args

def parse_losses(losses):
    log_vars = dict()
    for loss_name, loss_value in losses.items():
        log_vars[loss_name] = fluid.layers.mean(loss_value)

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss

    return loss, log_vars

args = parse_args()
config = parse_config(args.config)
train_config = merge_configs(config, 'train', vars(args))
val_config = merge_configs(config, 'valid', vars(args))
train_reader = KineticsReader(args.model_name.upper(), 'train', train_config).create_reader()
val_reader = KineticsReader(args.model_name.upper(), 'valid', val_config).create_reader()

label = fluid.layers.data(name='label', shape=[1], dtype='int64')
data_shape = [1, 3, 32, 224, 224]
img = fluid.layers.data(name='images', shape=data_shape, dtype='float32')

network = TSN.TSN3D(backbone=train_config['MODEL']['backbone'],
                                necks=train_config['MODEL']['necks'],
                                spatial_temporal_module=train_config['MODEL']['spatial_temporal_module'],
                                segmental_consensus=train_config['MODEL']['segmental_consensus'],
                                cls_head=train_config['MODEL']['cls_head'])
losses, acc = network.net(img, label, is_training_val=True)
loss, log_vars = parse_losses(losses)

# 获取测试程序
test_program = fluid.default_main_program().clone(for_test=True) 

boundaries = [100 * 5000, 125 * 5000]  # 7500
values = [0.01, 0.001, 0.0001]
lr = fluid.layers.piecewise_decay(boundaries=boundaries, values=values)
optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True,
                                        regularization=fluid.regularizer.L2Decay(1e-4))
optimizer.minimize(loss)
print("完成")

# 定义使用CPU还是GPU，使用CPU时use_cuda = False,使用GPU时use_cuda = True
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
parallel_places = [fluid.CUDAPlace(0), fluid.CUDAPlace(1), fluid.CUDAPlace(2), fluid.CUDAPlace(3)] if use_cuda else [fluid.CPUPlace()] * 4

# 创建执行器，初始化参数
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

compiled_train_prog = fluid.CompiledProgram(
    fluid.default_main_program()).with_data_parallel(
            loss_name=loss.name, places=parallel_places)
            
# compiled_test_prog = fluid.CompiledProgram(
#     test_program).with_data_parallel(
#             share_vars_from=compiled_train_prog,
#             places=parallel_places)

#定义数据映射器
feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

#加载模型
# path = "tpnmodel/tpn_1"
# # fluid.io.load_persistables(exe, path, fluid.default_startup_program())
# fluid.io.load_persistables(exe, path, fluid.default_main_program())

#训练并保存模型    
EPOCH_NUM = 1  # 20  

for pass_id in range(EPOCH_NUM): 
    # 开始训练
    for batch_id, data in enumerate(train_reader()):                        #遍历train_reader的迭代器，并为数据加上索引batch_id
        train_lost, train_loss_cls, train_loss_aux, train_acc_cls,\
            train_acc_aux = exe.run(program=compiled_train_prog, #运行主程序 fluid.default_main_program()
                             feed=feeder.feed(data),                        #喂入一个batch的数据
                             fetch_list=[loss, log_vars['loss_cls'], log_vars['loss_aux'],\
                              acc['acc_cls'], acc['acc_aux']])                    #fetch均方误差和准确率
        
        #每100次batch打印一次训练、进行一次测试
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, loss:%0.5f, loss_cls:%0.5f, loss_aux:%0.5f, acc_cls:%0.5f, acc_aux:%0.5f' % (
                    pass_id, batch_id,
                    train_lost[0], train_loss_cls[0], train_loss_aux[0],
                    train_acc_cls[0], train_acc_aux[0]))

    path = output_dir+"/tpn_" + str(pass_id)
    if not os.path.exists(path):
        os.makedirs(path)
    fluid.io.save_persistables(exe, path, fluid.default_main_program()) # 用于继续训练或增量训练    

    # 开始测试
    test_cls_losts = []                                                         #测试的损失值
    test_cls_accs = []                                                          #测试的准确率
    for batch_id, data in enumerate(val_reader()):
        test_lost, test_loss_cls, test_acc_cls = exe.run(program=test_program,   #执行测试程序 compiled_test_prog
                                      feed=feeder.feed(data),               #喂入数据
                                      fetch_list=[loss, log_vars['loss_cls'], acc['acc_cls']])           #fetch 误差、准确率
        test_cls_losts.append(test_loss_cls[0])                                     #记录每个batch的误差
        test_cls_accs.append(test_acc_cls[0])                                       #记录每个batch的准确率

    # 求测试结果的平均值
    test_cls_lost = (sum(test_cls_losts) / len(test_cls_losts))                         #计算误差平均值（误差和/误差的个数）
    test_cls_acc = (sum(test_cls_accs) / len(test_cls_accs))                            #计算准确率平均值（ 准确率的和/准确率的个数）
    print('Test:%d, test_cls_lost:%0.5f, test_cls_acc:%0.5f' % (pass_id, test_cls_lost, test_cls_acc))
    
