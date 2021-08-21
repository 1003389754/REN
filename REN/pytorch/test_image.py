import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
from loss import SoftEntropy, CrossEntropyLabelSmooth
import pre_process as prep
from torch.utils.data import DataLoader
from tool import image_classification_test, mse_with_softmax,sigmoid_rampup,update_ema_variables
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
import time
from datetime import datetime
from visualdl import LogWriter
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torch.nn import functional as F
from collections import OrderedDict
from torchstat import stat
def test(config):
    ## set pre-process 设置导入数据格式
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=8, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=8, drop_last=True)

    if prep_config["test_10crop"]:  # 测试图片进行十次的裁剪
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                       transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                               shuffle=False, num_workers=8) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                          shuffle=False, num_workers=8)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    # base_network = net_config["name"](**net_config["params"]).cuda()
    base_network = net_config["name"](**net_config["params"])
    # for name, param in base_network.named_parameters():
    #     print(name, param.shape)
    stat(base_network,(3,227,227))
    # gpus = config['gpu'].split(',')
    # if len(gpus) > 1:
    #     base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
    # #cudnn.benchmark = True
    # dict = torch.load(config["model_path"])
    # newdict = OrderedDict()
    # for key in dict.keys():
    #     #print(dict.keys())
    #     #print(key[2:])
    #     newdict[key[2:]]=dict[key]
    # #print(dict)
    # #print(newdict)
    #
    # base_network.load_state_dict(newdict)
    # # base_network.load_state_dict(
    # #     {k.replace('module.', ''): v for k, v in torch.load(config["model_path"]).items()})
    # #base_network = base_network.cuda(device_ids[0])
    # #print("base_network",base_network)
    # #print(base_network.get_parameters)
    # # print(base_network.state_dict())
    # # for key,v in base_network.state_dict():
    # #     print(key,v)
    #
    # ## test
    # len_train_source = len(dset_loaders["source"])
    # len_train_target = len(dset_loaders["target"])
    # best_acc = 0.0
    # source_test=True
    # for i in range(config["num_iterations"]):
    #     start = time.time()
    #     base_network.train(False)
    #     temp_acc_s,temp_acc_t = image_classification_test(config["output_path"], dset_loaders, base_network, i,
    #                                                       prep_config["test_10crop"], True)
    #
    #     if temp_acc_t > best_acc:
    #         best_acc = temp_acc_t
    #     end = time.time()
    #     log_str = "Testing:epoch: [{}/{}], temp_acc_s: {:.2%},temp_acc_t: {:.2%}, test_best_pre: {:.2%} time:{:2f}" \
    #         .format(i, config["num_iterations"], temp_acc_s,temp_acc_t, best_acc, (end - start) / 60)
    #     config["out_file"].write(log_str + "\n")
    #     config["out_file"].flush()
    #     print(log_str)
    # return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50',
                        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13",
                                 "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'],
                        help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../../data/office/webcam_10_list.txt',
                        help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../../data/office/amazon_31_list.txt',
                        help="The target dataset path list")
    parser.add_argument('--model_dir', type=str, default='san',
                        help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--output_path', type=str, default='san', help="output directory path")
    parser.add_argument('--log', type=str, default='log', help="visualdll log name")
    parser.add_argument('--batchsize', type=int, default='32', help="batchsize")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    dt = datetime.now()
    local_time = str(dt.year) + "_" + str(dt.month) + "_" + str(dt.day)
    # print(local_time)
    # train config
    config = {}
    config["gpu"] = args.gpu_id
    config["batchsize"] = args.batchsize
    config["num_iterations"] = 2
    config["output_path"]=args.model_dir
    config["model_path"]=args.model_dir +"/best_model.pth.tar"

    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "test.txt"), "w")
    # print(config["out_file"])
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop": True, 'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }

    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 64}, \
                      "target": {"list_path": args.t_dset_path, "batch_size": config["batchsize"]}, \
                      "test": {"list_path": args.t_dset_path, "batch_size": 64}}
    if config["dataset"] == "office":
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "image-clef":
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "office-home":
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()
    print(config)
    starttime = time.time()
    test(config)
    endtime = time.time()
    dtime = endtime - starttime
    print("Runingtime:{:2f}m".format(dtime / 60))
