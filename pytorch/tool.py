import os.path as osp

import numpy as np
import torch
import torch.nn as nn

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torch.nn import functional as F

def image_classification_test(out_path, loader, model, epoch, test_10crop=True, source_test=True):
    #target test
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output_t = outputs.float().cpu()
                    all_label_t = labels.float().cpu()
                    start_test = False
                else:
                    all_output_t = torch.cat((all_output_t, outputs.float().cpu()), 0)
                    all_label_t = torch.cat((all_label_t, labels.float().cpu()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output_t = outputs.float().cpu()
                    all_label_t = labels.float().cpu()
                    start_test = False
                else:
                    all_output_t = torch.cat((all_output_t, outputs.float().cpu()), 0)  # cat拼接，按第0维度拼接
                    all_label_t = torch.cat((all_label_t, labels.float().cpu()), 0)
    _, predict_t = torch.max(all_output_t, 1)  # 按第1维度求解最大值，返回的第二个值pred为索引值
    accuracy_t = torch.sum(torch.squeeze(predict_t).float() == all_label_t.cpu()).item() / float(all_label_t.size()[0])  # squeeze去除size为1的维度
    showonT_tsne(out_path,all_output_t,epoch,all_label_t)

    accuracy_s=0
    if source_test:
        # source test
        start_test = True
        with torch.no_grad():
            iter_test = iter(loader["source"])
            for i in range(len(loader['source'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output_s = outputs.float().cpu()
                    all_label_s = labels.float().cpu()
                    start_test = False
                else:
                    all_output_s = torch.cat((all_output_s, outputs.float().cpu()), 0)  # cat拼接，按第0维度拼接
                    all_label_s = torch.cat((all_label_s, labels.float().cpu()), 0)

        _, predict_s = torch.max(all_output_s, 1)  # 按第1维度求解最大值，返回的第二个值pred为索引值
        accuracy_s = torch.sum(torch.squeeze(predict_s).float() == all_label_s.cpu()).item() / float(all_label_s.size()[0])  # squeeze去除size为1的维度
        len_s = len(all_label_s)
        outputs = torch.cat((all_output_s, all_output_t), dim=0)
        all_label = torch.cat((all_label_s,all_label_t), dim=0)
        showall_tsne(out_path, outputs, epoch, len_s)
        showT_tsne(out_path, outputs, epoch,len_s,all_label)
        #print("Source_test_acc:",accuracy_s)

    return accuracy_s,accuracy_t

def showall_tsne(out_path, data, epoch,len_s):
    embeddings = TSNE(n_components=2, learning_rate=1000, init='pca', random_state=0).fit_transform(data)
    vis_x = embeddings[:len_s, 0]
    vis_y = embeddings[:len_s, 1]
    plt.scatter(embeddings[:len_s, 0], embeddings[:len_s, 1], s=20, c='r', marker='.')
    plt.scatter(embeddings[len_s:, 0], embeddings[len_s:, 1], s=20, c='b', marker='.')
    #plt.colorbar()
    #plt.clim(-0.5, 31.5)
    pltpatch = out_path + "/" + str(epoch) + 'test.png'
    plt.savefig(pltpatch, dpi=300)
    plt.close()
    # plt.show()

def showT_tsne(out_path,data,epoch,len_s,label=None):
    print("label",label.size())
    print("data",data.size())
    embeddings = TSNE(n_components=2,learning_rate=1000,init = 'pca',random_state=0).fit_transform(data)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    #print(len(vis_x),len(vis_y))
    plt.scatter(vis_x, vis_y, s=20, c=label, marker='.',cmap=plt.cm.get_cmap("jet"))

    # plt.scatter(embeddings[:len_s, 0], embeddings[:len_s, 1], s=20, c=label[:len_s], marker='.',
    # cmap=plt.cm.get_cmap("jet")) plt.scatter(embeddings[len_s:, 0], embeddings[len_s:, 1], s=20, c=label[:len_s],
    # marker='.',cmap=plt.cm.get_cmap("jet")) #plt.colorbar() plt.clim(-0.5, 31.5)
    pltpatch = out_path+"/"+str(epoch)+'stu.png'
    if osp.exists(pltpatch):
        pltpatch = out_path+"/"+str(epoch)+'tea.png'
    plt.savefig(pltpatch,dpi=300)
    plt.close()
    #plt.show()
def showonT_tsne(out_path,data,epoch,label=None):
    print(label.size())
    embeddings = TSNE(n_components=2,learning_rate=1000,init = 'pca',random_state=0).fit_transform(data)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    #print(len(vis_x),len(vis_y))
    plt.scatter(vis_x, vis_y, s=40, c=label, marker='.',cmap=plt.cm.get_cmap("jet"))
    #plt.colorbar()
    # plt.clim(-0.5, 31.5)
    pltpatch = out_path+"/"+str(epoch)+'onTstu.png'
    if osp.exists(pltpatch):
        pltpatch = out_path+"/"+str(epoch)+'onTtea.png'
    plt.savefig(pltpatch,dpi=300)
    plt.close()
    #plt.show()
def find_pseudo(outputs,label):
    softmax_out_t = nn.Softmax(dim=1)(outputs)
    #print(softmax_out_t.size())
    # print(softmax_out_t)
    #print(torch.max(softmax_out_t, 1))
    values_t, predict = torch.max(softmax_out_t, 1)
    acc_t = torch.sum(torch.squeeze(predict).float() == label.cuda()).item() / float(label.size()[0])
    #print(label)
    labels_eq = torch.eq(predict, label.cuda())
    #print(labels_eq)
    print(acc_t)
    print("True", end=":")
    for j in range(len(labels_eq)):
        if labels_eq[j] == True:
            print(round(values_t[j].item(), 3), end=" ")
    print("\n")
    print("False", end=":")
    for j in range(len(labels_eq)):
        if labels_eq[j] == False:
            if values_t[j].item() >= 0.5:
                print(round(values_t[j].item(), 3), end=" ")
    print("\n")

def mse_with_softmax(logit1, logit2):
    assert logit1.size()==logit2.size()
    return F.mse_loss(F.softmax(logit1,1), F.softmax(logit2,1))

def sigmoid_rampup(epoch,total_epochs):
    if epoch>=total_epochs:
        return 1.0
    elif epoch<=0:
        return 0.0
    else:
        phase = 1.0 -float(epoch) / total_epochs
        return float(np.exp(-5.0*phase*phase))

def update_ema_variables(model,ema_model,alpha,global_step):
    alpha = min(1-1/(global_step+1),alpha)
    for ema_param,param in zip(ema_model.parameters(),model.parameters()):
        ema_param.data.mul_(alpha).add_(1-alpha,param.data)
