from plot_utils import *
import pickle
import matplotlib.pyplot as plt

l_all=[]
step_list=[]
batch_size_list=[]

list_all=[]
dataset_name='CIFAR10'
batch_size=128
dir_name='result/'+dataset_name+'/test/'
algo_list=['SGD','SGD+Armijo']
c_list=[0.1,0.01]


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
    

