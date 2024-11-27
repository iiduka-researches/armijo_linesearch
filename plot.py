from plot_utils import *
import pickle
import matplotlib.pyplot as plt
import argparse

l_all=[]
step_list=[]
batch_size_list=[]

parser = argparse.ArgumentParser()  
parser.add_argument('--dataset',type=str,help='dataset',default='CIFAR10')
parser.add_argument('--dir', type=str,help='directly',default='test') 
parser.add_argument('--batch',type=int,help='batch size',default='128')

args = parser.parse_args()     


dataset_name=args.dataset
dir_name=args.dir
batch_size=args.batch


algo_list=['SGD','SGD+Armijo']
c_list=[0.1,0.01]
dir_name='result/'+dataset_name+'/'+dir_name+'/'

for algo in algo_list:
    if 'Armijo' in algo:
        for c in c_list:
            file_name=dir_name+dataset_name+'_'+algo+str(c)+'_'+str(batch_size)+'.bin'
            print(file_name)
            with open(file_name,'rb') as p:
                l=pickle.load(p)
            l['algorithm']=l['algorithm']+str(c)
            l_all+=[l]
            print(l.keys())
    else:
        file_name=dir_name+dataset_name+'_'+algo+'_'+str(batch_size)+'.bin'
        print(file_name)
        with open(file_name,'rb') as p:
            l=pickle.load(p)
        l_all+=[l]
        print(l.keys())

plot_results(l_all,dir_name,dataset_name)
    

