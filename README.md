### Prepare project to code something:

#### Install uv
```shell
irm https://astral.sh/uv/install.ps1 | iex
```

#### Install venv
```shell
uv sync
```

#### Update your terminal to use virtual env

#### Install pre-commit hooks
```shell
pre-commit install
```

#### Update pre-commit hooks
```shell
pre-commit autoupdate
```

#### Some instructions:
##### To install some libraries, you need to use this command:
```shell
uv add <your_lib_name>
```
##### To do some customizations in venv, use only uv commands, check out this documentation:
https://docs.astral.sh/uv/