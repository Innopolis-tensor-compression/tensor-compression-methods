# Instructions for developers:

Use Linux or macOS. (cause tensorflow moment).

pyproject.toml  set uped for ubuntu-22.04 on WSL.

In this project, Python 3.11 was used.

## Main instruction with uv tool (recommended)

In this project UV Project Manager was used, check its [documentation](https://docs.astral.sh/uv) and [source code](https://github.com/astral-sh/uv).

### Install uv

#### On macOS and Linux.

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### On Windows.

```shell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### With pip.

```shell
pip install uv
```

See the [installation documentation](https://docs.astral.sh/uv/getting-started/installation/) for details and alternative installation methods.

### Install venv by [sync command](https://docs.astral.sh/uv/reference/cli/#uv-sync)

```shell
uv sync
```

### Activate venv

#### On macOS and Linux.

```
source .venv/bin/activate
```

#### On Windows.

```
.venv\Scripts\activate
```

##### Check GPU available for torch

```shell
python -c "import torch; print(torch.cuda.is_available())"
```

##### Check GPU available for tensorflow

```shell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### Set up the git hook scripts by [pre-commit](https://pre-commit.com/#3-install-the-git-hook-scripts)

```shell
pre-commit install
```

#### (Optional) Update pre-commit hooks

```shell
pre-commit autoupdate
```