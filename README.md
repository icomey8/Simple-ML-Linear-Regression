# Simple-ML-Model-Linear-Regression

This is a Python script that generates a random dataset of 100 items, builds and trains a model to predict the linear trend of the last portion of the dataset, and visualizes those predictions on a graph.  Pytorch, Matplotlib, time, and pathlib were used.

The code can be downloaded and ran with little setup.  If Pytorch has not been installed, run the following command:
```python
pip3 install torch torchvision
```

**Note** - The script will generate the same dataset each time, which is intentional.  This is because of `torch.manual_seed(100)` (Line 8 of `Train_Model.py`), which ensures that the same dataset is generated each time for reproducibility purposes.  The number inside the function can be changed if desired.

