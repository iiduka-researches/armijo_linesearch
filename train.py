import os
import sys
from torch import optim
import torch
import math
import train_utils as tu
import pickle
import argparse  
from train_utils import train_net
from optimizers import sls
import numpy as np
import os
from datasets import get_datasets
from models import get_models
import torch.nn as nn
import random

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(os.getpid())

parser = argparse.ArgumentParser()  
parser.add_argument('--dataset',type=str,help='dataset',default='CIFAR10')
parser.add_argument('--dir', type=str,help='directly',default='test') 
parser.add_argument('--cuda',type=str,help='gpu',default='7')
parser.add_argument('--batch',type=int,help='batch size',default='128')
parser.add_argument('--model',type=str,help='model',default='resnet34')

args = parser.parse_args()     

dir_name='result/'+args.dataset+'/'+args.dir
print(f'dir_name: {dir_name}')
if not os.path.isdir('result'):
    os.mkdir('result')
if not os.path.isdir('result/'+args.dataset):
    os.mkdir('result/'+args.dataset)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

device = 'cuda:'+args.cuda
dataset_name=args.dataset
model=args.model
batch_size=args.batch
weight_decay=0
train_set,test_set=get_datasets(dataset_name)
epoch=5

alg_list=['SGD','SGD+Armijo']
c_list=[0.1,0.01]

for alg_name in alg_list:
    print(dataset_name)
    
    if 'Armijo' in alg_name:
        for c in c_list:

            net=get_models(dataset_name)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net.to(device)
            n_batches_per_epoch=math.ceil(len(train_set)/batch_size)
            l=train_net(dataset_name,net,train_set,test_set,optimizer=sls.SGD(device,net.parameters(),c=c,n_batches_per_epoch=n_batches_per_epoch,),epoch=epoch,device=device,alg_name=alg_name,batch_size=batch_size,)

            file_name=tu.get_file_name(dataset_name,batch_size,alg_name+str(c))
            print(file_name)
            with open(dir_name+'/'+file_name,'wb')as p:
                pickle.dump(l,p)

            del net
            torch.cuda.empty_cache()
    else:
            
        net=get_models(dataset_name)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        
        if 'Momentum' in alg_name:
            l=train_net(dataset_name,net,train_set,test_set,optimizer=optim.SGD(net.parameters(),lr=0.1,momentum=0.9,weight_decay=0.000,nesterov=True),epoch=epoch,device=device,alg_name=alg_name,batch_size=batch_size)
        elif alg_name=='SGD':
            l=train_net(dataset_name,net,train_set,test_set,optimizer=optim.SGD(net.parameters(),lr=1,),epoch=epoch,device=device,alg_name=alg_name,batch_size=batch_size)
        elif alg_name=='Adam':
            l=train_net(dataset_name,net,train_set,test_set,optimizer=optim.Adam(net.parameters(),lr=1e-3,weight_decay=0),epoch=epoch,device=device,alg_name=alg_name,batch_size=batch_size)
        elif alg_name=='RMSProp':
            l=train_net(dataset_name,net,train_set,test_set,optimizer=optim.RMSprop(net.parameters(),lr=1e-3,weight_decay=weight_decay),epoch=epoch,device=device,alg_name=alg_name,batch_size=batch_size)
        elif alg_name=='AdamW':
            l=train_net(dataset_name,net,train_set,test_set,optimizer=optim.AdamW(net.parameters(),lr=1e-3,weight_decay=weight_decay),epoch=epoch,device=device,alg_name=alg_name,batch_size=batch_size)
        else:
            print(alg_name+'is not registered')
            sys.exit(1
            )
        del net
    file_name=tu.get_file_name(dataset_name,batch_size,alg_name)
    print(file_name)
    with open(dir_name+'/'+file_name,'wb')as p:
        pickle.dump(l,p)


    torch.cuda.empty_cache()
