# local-gemma-2
Gemma 2 optimized for your local machine

![image](https://github.com/huggingface/local-gemma-2/assets/12240844/b998347f-f481-4986-9a05-764420c69350)


## Instalation

```
pipx install .
```

> TODO: move to pip package. Has to be a local install for now, since it is a private repo

`pipx` creates an isolated Python evironment for the package. See their simple [installation instructions](https://github.com/pypa/pipx?tab=readme-ov-file#install-pipx) if you need to install it.

Alternativelly, you can also install on your Python environment through

```
pip install local-gemma-2
```

## Python Usage

```python
from local_gemma_2 import LocalGemma2ForCausalLM, AutoTokenizer

# TODO(SG): update model and API before release
model = LocalGemma2ForCausalLM.from_pretrained("fxmarty/tiny-random-GemmaForCausalLM", preset="speed")
tokenizer = AutoTokenizer.from_pretrained("fxmarty/tiny-random-GemmaForCausalLM")

prompt_ids = tokenizer("The cat sat on the mat", return_attention_mask=True, return_tensors="pt")
gen_ids = model.generate(**prompt_ids)

gen_text = tokenizer.batch_decode(gen_ids)
```

## CLI Usage

```
local-gemma-2
```

Call `local-gemma-2 -h` for available options.
