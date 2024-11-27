import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size': 18})
plt.rc('font', family='serif')
plt.style.use('seaborn-muted')

def plot_results(list_results,dir_name,dataset_name):

    Name_tilte = dataset_name

    ColorsList = ['b', 'r', 'c', 'g', 'y', 'k', 'm', 'brown','purple']

    for i in range(len(list_results)):
        list_results[i]['Color'] = ColorsList[i]

    fig, ax = plt.subplots(figsize =  (16,8))
    fig.patch.set_facecolor('white')

    for dict_save in list_results:
        ax.plot(dict_save['epoch_list'], dict_save['train_acc_list'], label= dict_save['algorithm'], color = dict_save['Color'], linewidth=3)

    # ax.grid(True)
    lgd = plt.legend(frameon=True, loc = 'upper right', framealpha = 1, edgecolor = 'black', fancybox = False,)

    lgd.get_frame().set_linewidth(1.0)
    for line in lgd.get_lines():
        line.set_linewidth(3.0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))

    ax.set_ylabel('Training loss')
    ax.set_xlabel('Epochs')

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.grid()
    ax.xaxis.set_tick_params(top='on', direction='in', width=2)
    ax.yaxis.set_tick_params(right='on', direction='in', width=2)
    if dataset_name=='CIFAR100':
        ax.set_ylim([0,1]) 
    
    elif dataset_name=='CIFAR10':
        ax.set_ylim([0.5,1]) 
    elif dataset_name=='r_MNIST':
        ax.set_ylim([0.90,0.99]) 

    plt.title('', fontsize = 50, fontweight='normal')
    plt.yscale("linear")
    plt.savefig(dir_name+'/train_acc.png')
    plt.savefig(dir_name+'/train_acc.pdf')
    plt.show()
    plt.clf()
    plt.close()


    fig, ax = plt.subplots(figsize =  (16,8))
    fig.patch.set_facecolor('white')

    for dict_save in list_results:
        ax.plot(dict_save['epoch_list'], dict_save['train_loss_list'], label= dict_save['algorithm'], color = dict_save['Color'], linewidth=3)

    ax.grid(True)
    lgd = plt.legend(frameon=True, loc = 'upper right', framealpha = 1, edgecolor = 'black', fancybox = False)
    lgd.get_frame().set_linewidth(1.0)
    for line in lgd.get_lines():
        line.set_linewidth(3.0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels),)

    ax.set_ylabel('Training loss')
    ax.set_xlabel('Epochs')

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    ax.xaxis.set_tick_params(top='on', direction='in', width=2)
    ax.yaxis.set_tick_params(right='on', direction='in', width=2)


    plt.title('', fontsize = 50, fontweight='normal')
    plt.yscale("log")
    plt.savefig(dir_name+'/train_loss.png')
    plt.savefig(dir_name+'/train_loss.pdf')
    plt.show()
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(figsize = (16,8))
    fig.patch.set_facecolor('white')

    for dict_save in list_results:
        ax.plot(dict_save['epoch_list'], dict_save['test_acc_list'], label= dict_save['algorithm'], color = dict_save['Color'], linewidth=3)

    # ax.grid(True)
    lgd = plt.legend(frameon=True, loc = 'upper right', framealpha = 1, edgecolor = 'black', fancybox = False)
    lgd.get_frame().set_linewidth(1.0)
    for line in lgd.get_lines():
        line.set_linewidth(3.0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(bbox_to_anchor=(1, 0), loc='lower right')

    ax.set_ylabel('Test Accuracy')
    ax.set_xlabel('Epochs')

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.grid()
    ax.xaxis.set_tick_params(top='on', direction='in', width=2)
    ax.yaxis.set_tick_params(right='on', direction='in', width=2)
  
    if dataset_name=='CIFAR100':
        ax.set_ylim([0.4,0.75]) 
        print(dataset_name)
    elif dataset_name=='CIFAR10':
        ax.set_ylim([0.0,1]) 
    elif dataset_name=='r_MNIST':
        ax.set_ylim([0.90,0.99]) 

    ax.set_yticks(np.arange(0.4, 0.75, 0.05)) 
    plt.title('', fontsize = 50, fontweight='normal')
    plt.yscale("linear")
    plt.savefig(dir_name+'/test_acc.png')
    plt.savefig(dir_name+'/test_acc.pdf')

    plt.show()
    plt.clf()
    plt.close()
