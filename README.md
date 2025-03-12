### Usage
Use linesearch in your code by adding the following script.

```python
import optimizers.sls
for i, (xx, yy) in enumerate(train_loader):
  # create loss closure
  closure = lambda: nn.CrossEntropyLoss(reduction='mean')(net(xx), yy)
  # update parameters
  optimizer.zero_grad()
  optimizer.step(closure=closure)
```

#### Install
```
pip install -r requirements.txt
```

#### training
```
python train.py --cuda 0 --batch 128 --dataset CIFAR100 --dir test
```
#### plot
```
python plot.py
```
