# EvidentialCDM
Pytorch implementation for Learning Evidential Cognitive Diagnosis Networks Robust to Response Bias

  Train the model:

python train.py {device} {epoch}

  For example:

python train.py cuda:0 10 or python train.py cpu 10

  Test the trained the model on the test set:

python predict.py {epoch}
