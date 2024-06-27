# local-gemma-2
Gemma 2 optimized for your local machine

![image](https://github.com/huggingface/local-gemma-2/assets/12240844/b998347f-f481-4986-9a05-764420c69350)


## Instalation

```
pipx install .
```

> TODO: move to pip package. Has to be a local install for now, since it is a private repo

`pipx` creates an isolated Python evironment for the package. See their installation instructions [here](https://github.com/pypa/pipx?tab=readme-ov-file#install-pipx).

Alternativelly, you can also install on your Python environment through

```
pip install local-gemma-2
```

### Docker Installation

> TODO(SG): add installation

## Usage

You can chat with the model through an interactive session by calling

```
local-gemma-2
```

Alternativelly, you can request an output by passing the prompt, such as

```
local-gemma-2 What is the capital of France?
```

Depending on your device and setup, a few optimizations will be automatically enabled (e.g. Flash Attention). You can select different types of optimization through the `--optimization` flag, e.g. `memory` for GPU memory constrained situations. You can also control the style of the generated text through the `--mode` flag, e.g. `factual` to minimize halluciations. Here's an example:

```
local-gemma-2 --optimization memory --mode creative
```

Call `local-gemma-2 -h` for all available options.
