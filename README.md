# Learning from Similarity-Confidence Data

## Requirements
- Python 3.6
- numpy 1.14
- PyTorch 1.1
- torchvision 0.2

## Demo
The following demo will show the results of Sconf Learning with the MNIST dataset. When running the code, the test accuracy of each epoch will be printed for `u`, `abs`, `nn`, and `sd`. The results will have two columns: epoch number and test accuracy. For `siamese` and `contrastive`. the test accuracy will be printed after the end of training.

```bash
python demo.py -me  u    (the method is optional as shown below)
```

#### Methods
Before running `demo.py`, we have to choose one of the 5 methods available:

- `u`: Learning with unbiased Sconf risk estimator.
- `abs`: Learning with Sconf-ABS estimator.
- `nn`: Learning with Sconf-NN estimator.
- `sd`: Classification from Pairwise Similarities/Dissimilarities via Empirical Risk Minimization.
- `siamese`: Learning from pairwise similarity using Siamese network.
- `contrastive`: Learning from pairwise similarity using Siamese network and contrastive loss.