import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
from loss import *
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

def train(config):
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

    if prep_config["test_10crop"]: #测试图片进行十次的裁剪
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
    base_network = net_config["name"](**net_config["params"])
    base_network_teacher = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    base_network_teacher = base_network_teacher.cuda()

    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    #set teacher parameters
    # for param in base_network_teacher.parameters():
    #     param.detach_()
    ce_loss = CrossEntropyLabelSmooth(class_num).cuda()
    se_loss = SoftEntropy().cuda()

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        base_network_teacher = nn.DataParallel(base_network_teacher, device_ids=[int(i) for i in gpus])

    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = CL_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    tem_acc_t = 0.0
    tem_acc_t_tea = 0.0
    with LogWriter(logdir="./visualdll_logs/"+config["log_path"]) as wirter:
        for i in range(config["num_iterations"]+1):
            if (i % config["test_interval"] == 0 or i==1):
                start = time.time()
                base_network.train(False)
                base_network_teacher.train(False)
                if(i%2000==0 or i==1):
                    temp_acc_stu_s,temp_acc_stu_t = image_classification_test(config["output_path"], dset_loaders, \
                                                             base_network, i, prep_config["test_10crop"],True)
                    temp_acc_tea_s,temp_acc_tea_t = image_classification_test(config["output_path"], dset_loaders, \
                                                             base_network_teacher, i, prep_config["test_10crop"],True)
                temp_acc_stu_s,temp_acc_stu_t = image_classification_test(config["output_path"],dset_loaders, \
                                                         base_network, i, prep_config["test_10crop"])
                temp_acc_tea_s,temp_acc_tea_t = image_classification_test(config["output_path"],dset_loaders, \
                                                         base_network_teacher, i, prep_config["test_10crop"])
                # temp_model = nn.Sequential(base_network)
                if temp_acc_stu_t > temp_acc_tea_t:
                    temp_acc = temp_acc_stu_t
                    temp_model = nn.Sequential(base_network)
                else:
                    temp_acc = temp_acc_tea_t
                    temp_model = nn.Sequential(base_network_teacher)
                if temp_acc > best_acc:
                    best_acc = temp_acc
                    best_model = temp_model
                end = time.time()
                log_str = "Testing:epoch: [{}/{}], stu_pre: {:.2%}, tea_pre: {:.2%}, test_pre: {:.2%} time:{:2f}" \
                    .format(i,config["num_iterations"], temp_acc_stu_t, temp_acc_tea_t, best_acc,(end-start)/60)
                config["out_file"].write(log_str + "\n")
                config["out_file"].flush()
                print(log_str)
                wirter.add_scalar(tag='Test/stu_pre', step=i, value=temp_acc_stu_t)
                wirter.add_scalar(tag='Test/tea_pre', step=i, value=temp_acc_tea_t)
            # if i % config["snapshot_interval"] == 0:
            #     torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
            #         "iter_{:05d}_model.pth.tar".format(i)))

            loss_params = config["loss"]
            ## train one iter
            base_network.train(True)
            base_network_teacher.train(True)
            ad_net.train(True)
            optimizer = lr_scheduler(optimizer, i, **schedule_param)
            optimizer.zero_grad()
            if i % len_train_source == 0:
                iter_source = iter(dset_loaders["source"])
            if i % len_train_target == 0:
                iter_target = iter(dset_loaders["target"])
            inputs_source, labels_source = iter_source.next() #labels_source.size=[32]
            inputs_target, labels_target = iter_target.next()
            bachsize_s = inputs_source.size(0)
            bachsize_t = inputs_target.size(0)

            inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
            features_source, outputs_source = base_network(inputs_source) #inputs_source.size=[32,3,224,224]
            features_target, outputs_target = base_network(inputs_target) #features_target.size=[32,256],outputs_target.size=[32,31]
            #print(outputs_target.size())
            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)
            #find_pseudo(outputs_target,labels_target)

            if i==0:
                ema_softmax_out = softmax_out
            else:
                alpha = min(1 / (i + 1), args.alpha)
                ema_softmax_out.data.mul_(alpha).add_(1-alpha,softmax_out.data)

            with torch.no_grad():
                features_source_tea, outputs_source_tea = base_network_teacher(inputs_source)
                features_target_tea, outputs_target_tea = base_network_teacher(inputs_target)
                outputs_source_tea = outputs_source_tea.detach()
                outputs_target_tea = outputs_target_tea.detach()

            features_tea = torch.cat((features_source_tea, features_target_tea), dim=0)
            outputs_tea = torch.cat((outputs_source_tea, outputs_target_tea), dim=0)
            softmax_out_tea = nn.Softmax(dim=1)(outputs_tea)
            #print("EMA:")
            #find_pseudo(outputs_target_tea,labels_target)

            if i==0:
                ema_softmax_out_tea = softmax_out_tea
            else:
                alpha = min(1 / (i + 1), args.alpha)
                ema_softmax_out_tea.data.mul_(alpha).add_(1-alpha,softmax_out_tea.data)

            if config['method'] == 'CDAN+E':
                if config['lossf'] == 'noema':
                    entropy = loss.Entropy(softmax_out)
                    transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i),random_layer)
                elif config['lossf'] == 'Sema':
                    entropy_s = loss.Entropy(ema_softmax_out)
                    transfer_loss_ema = loss.CDAN([features, ema_softmax_out], ad_net, entropy_s, network.calc_coeff(i), random_layer)
                elif config['lossf'] == 'Tema':
                    entropy_t = loss.Entropy(ema_softmax_out_tea)
                    transfer_loss_tea = loss.CDAN([features, ema_softmax_out_tea], ad_net, entropy_t, network.calc_coeff(i), random_layer)
                elif config['lossf'] == 'E':
                    transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i),random_layer)
                    transfer_loss_ema = loss.CDAN([features, ema_softmax_out], ad_net, entropy_s, network.calc_coeff(i), random_layer)
                else:
                    entropy = loss.Entropy(softmax_out)
                    entropy_s = loss.Entropy(ema_softmax_out)

                    transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i),random_layer)
                    transfer_loss_ema = loss.CDAN([features, ema_softmax_out], ad_net, entropy_s, network.calc_coeff(i), random_layer)
                    #transfer_loss_tea = loss.CDAN([features, ema_softmax_out_tea], ad_net, entropy, network.calc_coeff(i), random_layer)
                    transfer_loss_tea =0
            elif config['method'] == 'CDAN':
                if config['lossf'] == 'noema':
                    transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
                elif config['lossf'] == 'Sema':
                    transfer_loss_ema = loss.CDAN([features, ema_softmax_out], ad_net, None, None, random_layer)
                elif config['lossf'] == 'Tema':
                    transfer_loss_tea = loss.CDAN([features, ema_softmax_out_tea], ad_net, None, None, random_layer)
                elif config['lossf'] == 'cdan':
                    transfer_loss = loss.CDAN([features, ema_softmax_out], ad_net, None, None, random_layer)
                else:
                    transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
                    transfer_loss_ema = loss.CDAN([features, ema_softmax_out], ad_net, None, None, random_layer)
                    transfer_loss_tea = loss.CDAN([features, ema_softmax_out_tea], ad_net, None, None, random_layer)

            elif config['method'] == 'DANN':
                transfer_loss = loss.DANN(features, ad_net)
            else:
                raise ValueError('Method cannot be recognized.')

            con_loss = torch.div((mse_with_softmax(outputs_source,outputs_source_tea)*bachsize_s+
                        mse_with_softmax(outputs_target,outputs_target_tea)*bachsize_t)
                        ,(bachsize_t+bachsize_s))
            SE_loss = mse_with_softmax(outputs_target, outputs_target_tea)
            CL_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            MI_loss = MI(outputs_target)
            total_loss = SE_loss + CL_loss + 0.1*MI_loss
            if config['lossf']=='noema':
                total_loss += loss_params["trade_off"] * transfer_loss + con_loss*sigmoid_rampup(i,config["alpha_con"])
            elif config['lossf'] == 'Sema':
                total_loss += loss_params["trade_off"] * transfer_loss_ema + con_loss*sigmoid_rampup(i,config["alpha_con"])
                #total_loss = loss_params["trade_off"] * transfer_loss + CL_loss
            elif config['lossf'] == 'Tema':
                total_loss += loss_params["trade_off"] * transfer_loss_tea + con_loss * sigmoid_rampup(i, config["alpha_con"])
            elif config['lossf'] == 'E':
                total_loss += loss_params["trade_off"] * transfer_loss + 0.01*transfer_loss_ema + con_loss*sigmoid_rampup(i,config["alpha_con"])
            elif config['lossf'] == 'cdan':
                total_loss += loss_params["trade_off"] * transfer_loss
            else:
                total_loss += loss_params["trade_off"] * transfer_loss + 0.01*transfer_loss_tea + 0.01*transfer_loss_ema + con_loss*sigmoid_rampup(i,config["alpha_con"])
            _, predict = torch.max(outputs_source.float().cpu(), 1)
            acc_s = torch.sum(torch.squeeze(predict).float() == labels_source.cpu()).item() / float(labels_source.size()[0])
            _, predict = torch.max(outputs_source_tea.float().cpu(), 1)
            acc_s_tea = torch.sum(torch.squeeze(predict).float() == labels_source.cpu()).item() / float(labels_source.size()[0])
            _, predict = torch.max(outputs_target.float().cpu(), 1)
            acc_t = torch.sum(torch.squeeze(predict).float() == labels_target.cpu()).item() / float(labels_target.size()[0])
            _, predict = torch.max(outputs_target_tea.float().cpu(), 1)
            acc_t_tea = torch.sum(torch.squeeze(predict).float() == labels_target.cpu()).item() / float(labels_target.size()[0])
            tem_acc_t = max(acc_t,tem_acc_t)
            tem_acc_t_tea = max(acc_t_tea,tem_acc_t_tea)


            if config['lossf']=='noema':
                log_str = "Epoch:[{}/{}] tr_loss:{:.4f} CL_loss:{:.4f} " \
                          "SE_loss:{:.4f} con_loss:{:.4f} MI_loss:{:.4f} loss:{:.4f}" \
                    .format(i, config["num_iterations"], transfer_loss.item(),CL_loss.item(),
                            SE_loss.item(), con_loss.item() * 100, MI_loss.item(),total_loss.item())
                wirter.add_scalar(tag='Train/transfer_loss', step=i, value=transfer_loss.item())
            elif config['lossf'] == 'Sema':
                log_str = "Epoch:[{}/{}] tr_loss_ema:{:.4f} CL_loss:{:.4f} " \
                          "SE_loss:{:.4f} con_loss:{:.4f} loss:{:.4f}" \
                    .format(i, config["num_iterations"], transfer_loss_ema.item(),CL_loss.item(),
                            SE_loss.item(), con_loss.item() * 100, total_loss.item())
                wirter.add_scalar(tag='Train/transfer_loss_ema', step=i, value=transfer_loss_ema.item())
            elif config['lossf'] == 'Tema':
                log_str = "Epoch:[{}/{}]  tr_loss_tea:{:.4f} CL_loss:{:.4f} " \
                          "SE_loss:{:.4f} con_loss:{:.4f} loss:{:.4f}" \
                    .format(i, config["num_iterations"], transfer_loss_tea.item(),CL_loss.item(),
                            SE_loss.item(), con_loss.item() * 100, total_loss.item())
                wirter.add_scalar(tag='Train/transfer_loss_tea', step=i, value=transfer_loss_tea.item())
            elif config['lossf'] == 'E':
                log_str = "Epoch:[{}/{}] tr_loss_ema:{:.4f} CL_loss:{:.4f} " \
                          "SE_loss:{:.4f} con_loss:{:.4f} loss:{:.4f}" \
                    .format(i, config["num_iterations"], transfer_loss_ema.item(),
                            CL_loss.item(),
                            SE_loss.item(), con_loss.item() * 100, total_loss.item())
            elif config['lossf'] == 'cdan':
                log_str = "Epoch:[{}/{}] tr_loss:{:.4f} CL_loss:{:.4f} " \
                          "SE_loss:{:.4f} loss:{:.4f}" \
                    .format(i, config["num_iterations"], transfer_loss.item(),
                            CL_loss.item(),
                            SE_loss.item(), total_loss.item())
            else:
                log_str = "Epoch:[{}/{}] tr_loss_ema:{:.4f} tr_loss_tea:{:.4f} CL_loss:{:.4f} " \
                          "SE_loss:{:.4f} con_loss:{:.4f} loss:{:.4f}" \
                    .format(i, config["num_iterations"], transfer_loss_ema.item(), transfer_loss_tea.item(),CL_loss.item(),
                            SE_loss.item(), con_loss.item() * 100, total_loss.item())
                wirter.add_scalar(tag='Train/transfer_loss_ema', step=i, value=transfer_loss_ema.item())
                wirter.add_scalar(tag='Train/transfer_loss_tea', step=i, value=transfer_loss_tea.item())

            print(log_str)

            if (i % 100 == 0):
                config["out_file"].write(log_str + "\n")
                config["out_file"].flush()
            total_loss.backward()
            optimizer.step()
            update_ema_variables(base_network, base_network_teacher, args.alpha, i)
            wirter.add_scalar(tag='Train/classfier_loss', step=i, value=CL_loss.item())
            wirter.add_scalar(tag='Train/SE_loss', step=i, value=SE_loss.item())
            wirter.add_scalar(tag='Train/con_loss', step=i, value=con_loss.item())
            wirter.add_scalar(tag='Train/MI_loss', step=i, value=MI_loss.item())
            wirter.add_scalar(tag='Train/total_loss', step=i, value=total_loss.item())
    torch.save(best_model.state_dict(), osp.join(config["output_path"], "best_model.pth.tar"))
    log_str = "Testing:test_best_precision: {:.5f}".format(best_acc)
    print(log_str)
    config["out_file"].write(log_str + "\n")
    config["out_file"].flush()
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../../data/office/amazon_31_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../../data/office/webcam_10_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=100, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=2000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--output_path',type=str, default='san',help="output directory path")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--alpha',type=float,default=0.999,help="the alpha of mean teacher")
    parser.add_argument('--alpha_con',type=int,default=10000,help="the alpha of con loss")
    parser.add_argument('--log',type=str,default='log',help="visualdll log name")
    parser.add_argument('--batchsize', type=int, default='32', help="batchsize")
    parser.add_argument('--lossf',type=str,default='all')
    parser.add_argument('--num_iter', type=int, default=12000)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    if(args.dset=="office"):
        args.output_dir = args.s_dset_path[15:-9] + "2" + args.t_dset_path[15:-9]+"_"+args.method
    elif(args.dset=="image-clef"):
        args.output_dir = args.s_dset_path[19:-9] + "2" + args.t_dset_path[19:-9] + "_" + args.method
        #print(args.output_dir)
    elif (args.dset == "office-home"):
        x1=args.s_dset_path.rfind("/")+1
        #print(x1)
        x2 =args.t_dset_path.rfind("/")+1
        #print(x2)
        args.output_dir = args.s_dset_path[x1:-4] + "2" + args.t_dset_path[x2:-4] + "_" + args.method
        #print(args.output_dir)
    elif(args.dset == "visda"):
        args.output_dir = "visda"

    dt = datetime.now()
    local_time = str(dt.year)+"_"+str(dt.month)+"_"+str(dt.day)
    #print(local_time)
    # train config
    config = {}
    config["log_path"] = args.dset+'/'+args.output_path+args.output_dir+local_time+"_"+str(dt.hour)
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["batchsize"] = args.batchsize
    config["num_iterations"] = args.num_iter;
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["alpha_con"] = args.alpha_con
    if args.output_path == "san":
        log_dir = "logs/snapshot_"
    else:
        log_dir = "logs/"+args.output_path
    config["output_path"] = log_dir+"/"+ args.output_dir
    #print(config["output_path"])
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(log_dir, args.output_dir+".txt"), "w")
    #print(config["out_file"])
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":True, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":1.0}
    config["lossf"] = args.lossf
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
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":config["batchsize"]}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":config["batchsize"]}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":config["batchsize"]}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters       
        config["network"]["params"]["class_num"] = 31 
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config)+ "\n")
    config["out_file"].flush()
    print(config)
    starttime = time.time()
    train(config)


    endtime = time.time()
    dtime = endtime-starttime
    rtime="Runingtime:{:2f}h".format(dtime/3600)
    print(rtime)
    config["out_file"].write("\n"+rtime)
    config["out_file"].flush()
