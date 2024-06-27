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

## CLI Usage

```
local-gemma-2
```

Call `local-gemma-2 -h` for available options.

## Python Usage

Local Gemma-2 can be run locally through a Python interpreter using the familiar Transformers API. Local Gemma-2
provides four presets that trade-off accuracy, speed and memory[^1]:

| Mode           | Performance (?) | Inference Speed (tok/s) | Memory (GB) |
|----------------|-----------------|-------------------------|-------------|
| exact          |                 |                         |             |
| speed          |                 |                         |             |
| memory         |                 |                         |             |
| memory_extreme |                 |                         |             |

Based on the results above, you can select the preset that is most suited to your use-case. For example, if you require
the fastest inference, you should use the "speed" preset. Likewise, if you are constrained on memory, you should use 
either the "memory" or "memory_extreme" presets. 

To enable a preset, import the model class from the `local_gemma_2` package and pass the `preset` argument when 
loading the model `from_pretrained`. For example, the following code-snippet enables the "speed" preset for fastest 
inference:

```python
from local_gemma_2 import LocalGemma2ForCausalLM
from transformers import AutoTokenizer

# TODO(SG): update model and API before release
model = LocalGemma2ForCausalLM.from_pretrained("fxmarty/tiny-random-GemmaForCausalLM", preset="speed")
tokenizer = AutoTokenizer.from_pretrained("fxmarty/tiny-random-GemmaForCausalLM")

prompt_ids = tokenizer("The cat sat on the mat", return_attention_mask=True, return_tensors="pt")
gen_ids = model.generate(**prompt_ids.to(model.device))

gen_text = tokenizer.batch_decode(gen_ids)
```

### Preset Details

| Mode           | Attn Implementation | Torch Compile | Weights Dtype | CPU Offload |
|----------------|---------------------|---------------|---------------|-------------|
| exact          | eager               | yes           | fp16          | no          |
| speed          | sdpa                | yes           | fp16          | no          |
| memory         | sdpa                | yes           | int4          | no          |
| memory-extreme | sdpa                | no            | int2          | yes         |

---

[^1]: Benchmark performed using batch size 1 on an 80GB A100 GPU.