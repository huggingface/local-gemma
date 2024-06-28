# local-gemma-2
Gemma 2 optimized for your local machine

![image](https://github.com/huggingface/local-gemma-2/assets/12240844/b998347f-f481-4986-9a05-764420c69350)


## Installation

Local Gemma-2 provides `pipx` packages specific to your hardware. `pipx` creates an isolated Python environment for the 
package. See the simple [installation instructions](https://github.com/pypa/pipx?tab=readme-ov-file#install-pipx) if you need to install it.

### CUDA

```sh
pipx install ."[cuda]"
```

### MPS

```sh
pipx install ."[mps]"
```

### CPU

```sh
pipx install ."[cpu]"
```

> TODO: move to pip package. Has to be a local installation for now, since it is a private repo

<!---
```
pip install local-gemma-2
```
--->

### Docker Installation

> TODO(SG): add installation

## CLI Usage

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

## Python Usage

Local Gemma-2 can be run locally through a Python interpreter using the familiar Transformers API. Local Gemma-2
provides four presets that trade-off accuracy, speed and memory. The following table highlights this trade-off 
using [Gemma-2 9b](https://huggingface.co/google/gemma-2-9b) with batch size 1 on an 80GB A100 GPU:

| Mode           | Performance (?) | Inference Speed (tok/s) | Memory (GB) |
|----------------|-----------------|-------------------------|-------------|
| exact          | **a**           | 17.2                    | 18.3        |
| speed          | b               | **18.3**                | 18.3        |
| memory         | c               | 13.8                    | 7.3         |
| memory_extreme | d               | 7.0                     | **4.9**     |

To enable a preset, import the model class from the `local_gemma_2` package and pass the `preset` argument when 
loading the model `from_pretrained`. For example, the following code-snippet enables the "speed" preset for fastest 
inference with the Gemma-2 base model:

```python
from local_gemma_2 import LocalGemma2ForCausalLM
from transformers import AutoTokenizer

model = LocalGemma2ForCausalLM.from_pretrained("google/gemma-2-9b", preset="speed")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

model_inputs = tokenizer("The cat sat on the mat", return_attention_mask=True, return_tensors="pt")
generated_ids = model.generate(**model_inputs.to(model.device))

decoded_text = tokenizer.batch_decode(generated_ids)
```

When using an instruction-tuned model (prefixed by `-it`) for conversational use, prepare the inputs using a chat-template:

```python
from local_gemma_2 import LocalGemma2ForCausalLM
from transformers import AutoTokenizer

model = LocalGemma2ForCausalLM.from_pretrained("google/gemma-2-9b-it", preset="speed")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
model_inputs = encodeds.to(model.device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded_text = tokenizer.batch_decode(generated_ids)
```

### Minimum Memory Requirements

| Mode           | 9B   | 27B |
|----------------|------|-----|
| exact          | 18.3 |     |
| speed          | 18.3 |     |
| memory         | 7.3  |     |
| memory_extreme | 4.9  |     |

### Preset Details

| Mode           | Attn Implementation | Weights Dtype | CPU Offload |
|----------------|---------------------|---------------|-------------|
| exact          | eager               | fp16          | no          |
| speed          | sdpa                | fp16          | no          |
| memory         | sdpa                | int4          | no          |
| memory-extreme | sdpa                | int2          | yes         |
