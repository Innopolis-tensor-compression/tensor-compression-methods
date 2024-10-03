# Prepare project to code something:

## Instructions for developers:

In this project, Python 3.11 was used.

### Main instruction with uv tool (recommended)

In this project UV Project Manager was used, check its [documentation](https://docs.astral.sh/uv) and [source code](https://github.com/astral-sh/uv).

#### Install uv

```
# On macOS and Linux.
$ curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
$ powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
$ pip install uv
```

See the [installation documentation](https://docs.astral.sh/uv/getting-started/installation/) for details and alternative installation methods.

#### Install venv by [sync command](https://docs.astral.sh/uv/reference/cli/#uv-sync)

```shell
uv sync
```

```shell
.venv\Scripts\activate
```

```shell
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

(optional)
```shell
uv pip install tensorflow
```

#### Use created venv for next commands

#### Set up the git hook scripts by [pre-commit](https://pre-commit.com/#3-install-the-git-hook-scripts)

```shell
pre-commit install
```