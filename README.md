# Learning from Similarity-Confidence Data

## Requirements

- python 3.12
- pytorch 2.10.0
- torchaudio 2.10.0
- torchvision 0.25.0
- matplotlib

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

## Optional

### ruff, black, and pre-commit

This repository optionally supports linting and formatting with `ruff` and
`black`, managed through `pre-commit`.

If you want to enable them, install the optional dependency:

```bash
pip install -r requirements.txt
```

Then install the Git hook:

```bash
pre-commit install
```

After that, the configured hooks will run automatically on staged files before
each commit. At the moment, the hooks include:

- `end-of-file-fixer`
- `trailing-whitespace`
- `check-yaml`
- `check-toml`
- `ruff-check --fix`
- `black`

This project uses the configuration defined in `pyproject.toml`. Both `ruff`
and `black` use a line length of `120`.

If you want to run the formatter manually on all files, use:

```bash
pre-commit run --all-files
```

If you prefer to run the tools directly instead of through `pre-commit`, you
can use:

```bash
ruff check . --fix
black .
```

Using these tools is not required for running the experiments, but they help
keep the code style consistent across the repository.
