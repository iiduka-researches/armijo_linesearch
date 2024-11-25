import datetime
import time
from torchvision import transforms
import torchvision
from torch import optim
import torch
import torch.nn as nn
from torch.utils.data import(Dataset,DataLoader,TensorDataset)
import os
import io
import sys
import numpy as np
import math
import copy
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random



def get_full_grad_list(net,train_set,optimizer):
    parameters=[p for p in net.parameters()]
    batch_size=1000
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2)
    device='cuda:0'
    init=True
    full_grad_list=[]

    for i,(xx,yy)in (enumerate(train_loader)):
        xx = xx.to(device, non_blocking=True)
        yy = yy.to(device, non_blocking=True)
        h=net(xx)
        optimizer.zero_grad()
        loss=nn.CrossEntropyLoss(reduction='mean')(net(xx),yy)
        loss.backward()
        if init:
            for params in parameters:
                full_grad = torch.zeros_like(params.grad.detach().data)
                full_grad_list.append(full_grad)
            init=False

        for i,params in enumerate(parameters):
            g=params.grad.detach().data
            full_grad_list[i]+=(batch_size/len(train_set))*g
    full_grad_norm=compute_norm(full_grad_list,device=device)
    return full_grad_norm

def eval_net(net, dataset,batch_size, device):
    loader = DataLoader(dataset, drop_last=False, batch_size=1024)
    net.eval()

    with torch.no_grad(): 
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, y_pred = net(x).max(1)
            break 
    acc = (y == y_pred).float().sum() / len(y)
    
    return acc.item()

def compute_loss(net,dataset,device,batch_size):
    batch_size=max(batch_size,1024)
    loader = DataLoader(dataset, drop_last=False, batch_size=batch_size)
    net.eval()
    score_sum=0.
    for images,labels in (loader):
        images,labels=images.to(device),labels.to(device)
        score_sum+=nn.CrossEntropyLoss()(net(images),labels.view(-1)).item()
    score=float(score_sum/len(loader))
    return score

def compute_norm(param_list,device):
    param_norm=torch.tensor(0.,device=device)
    for p in param_list:
        if p is None:
            continue
        param_norm += torch.sum(torch.mul(p, p))
    param_norm = torch.sqrt(param_norm)
    return param_norm

def get_dir_name():
    dir_name='step_size_list'
    dir_count=0
    while(os.path.isdir('result/'+dir_name+'_'+str(dir_count))):
        if (sum(os.path.isfile(os.path.join('result/'+dir_name+'_'+str(dir_count),name)) for name in os.listdir('result/'+dir_name+'_'+str(dir_count))))==0:
            dir_name=dir_name+'_'+str(dir_count)
            return dir_name
        dir_count+=1
    dir_name=dir_name+'_'+str(dir_count)
    os.mkdir('result/'+dir_name)
    return dir_name

def get_file_name(dataset_name,batch_size,alg_name):
    file_name=dataset_name+'_'+alg_name+'_'+str(batch_size)+'.bin'
    return file_name


def eval_5_net(net,dataset,batch_size,device):
    loader = DataLoader(dataset, drop_last=False, batch_size=batch_size)
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.topk(outputs, 5, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.view(-1, 1)).sum().item()
            return correct/total



def train_net(dataset_name, net, train_set, test_set, optimizer, epoch, device, alg_name, batch_size):
    print(optimizer)
    optimizer_str = str(optimizer)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    train_loss_list = []
    train_acc = []
    test_acc = []
    test_acc_5 = []
    timesCPU = []
    epoch_list = []
    test_acc_list = []
    grad_norm_list = []
    step_size_list = []
    if_iter_zero = 0
    timesCPU_i = 0

    for e in range(epoch):
        
        if if_iter_zero == 0:
            start_time_wall_clock = time.time()
            start_time_cpu = time.process_time()
            timesCPU.append(0)
            if_iter_zero = 1

        net.train()
    
        for i, (xx, yy) in enumerate(train_loader):
            xx = xx.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)
            optimizer.zero_grad()
            closure = lambda: nn.CrossEntropyLoss(reduction='mean')(net(xx), yy)
            if 'SGD+Armijo' in alg_name:
                optimizer.step(closure=closure)
                step_size=optimizer.state['step_size']
                step_size_list.append(step_size)
                
            else:
                loss = nn.CrossEntropyLoss(reduction='mean')(net(xx), yy)
                loss.backward()
                optimizer.step()
                step_size = optimizer.param_groups[0]['lr']

        train_loss = compute_loss(net, train_set, device,batch_size)
        epoch_list.append(e)
        timesCPU_i = time.process_time() - start_time_cpu
        train_loss_list.append(train_loss)
        timesCPU.append(timesCPU_i)
        train_acc.append(eval_net(net, train_set,batch_size, device))
        test_acc.append(eval_net(net, test_set,batch_size, device))
        test_acc_5.append(eval_5_net(net, test_set,batch_size, device))

        ptrain_loss_list = '{:.5f}'.format(train_loss_list[-1])
        ptrain_acc = '{:.5f}'.format(train_acc[-1])
        ptest_acc = '{:.5f}'.format(test_acc[-1])
        ptest_acc_5 = '{:.5f}'.format(test_acc_5[-1])
        ptime = '{:.5f}'.format(timesCPU_i)
        pstep_size='{:.5f}'.format(step_size)
        
        if 'SGD+Armijo' in alg_name:
            print(f'e:{e},l:{ptrain_loss_list},t_acc{ptrain_acc},v_acc:{ptest_acc},test_acc_5:{ptest_acc_5},time:{ptime},step_size:{step_size},')
        else:
            print(f'e:{e},l:{ptrain_loss_list},t_acc{ptrain_acc},v_acc:{ptest_acc},test_acc_5:{ptest_acc_5},time:{ptime},')


    
    timesCPU=np.asarray(timesCPU)
    step_size_list=np.asarray(step_size_list)
    train_loss_list=np.asarray(train_loss_list)
    train_acc_list=np.asarray(train_acc)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    test_acc_list=np.asarray(test_acc)
    epoch_list=np.asarray(epoch_list)
    dict_result = {'algorithm': alg_name,
                   'dataset': dataset_name,
                   'train_loss_list': train_loss_list,
                   'train_acc_list': train_acc_list,
                   'test_acc_list': test_acc_list,
                   'timesCPU': timesCPU,
                   'epoch_list': epoch_list,
                   'step_size_list': step_size_list,}
    return dict_result


