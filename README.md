### Experiments

####Install the experiment requirements 
`pip install -r requirements.txt`

#### select algorithm
`python train.py --cuda 0 --batch 128 --dataset CIFAR100 --algorithm SGD+Armijo --dir 'your_directly'`

#### plot accuracy
`python plot_list/plot_accuracy --dir 'your_directly'`
