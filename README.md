<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local-gemma-2/local_gemma_2.png?raw=true" width="600"/>
</p>

<h3 align="center">
    <p>Run Gemma-2 locally in Python, fast!</p>
</h3>

This repository provides a lightweight wrapper around the [🤗 Transformers](https://github.com/huggingface/transformers)
library for easily running [Gemma-2](https://huggingface.co/blog/gemma2) on a local machine.

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

You can chat with the Gemma-2 through an interactive session by calling:

```sh
local-gemma-2
```

Alternatively, you can request an output by passing the prompt, such as:

```sh
local-gemma-2 What is the capital of France?
```

You can also pipe in other commands, which will be appended to the prompt after a `\n` separator

```sh
ls -la | local-gemma-2 decribe my files
```

By default, this loads the [Gemma-2 9b](https://huggingface.co/google/gemma-2-9b) model. To load the [Gemma-2 27b](https://huggingface.co/google/gemma-2-27b)
model, you can set the `--model` argument accordingly:

```sh
local-gemma-2 --model 27b
```

Local Gemma-2 will automatically find the most performant preset for your hardware, trading-off speed and memory. For more
control over generation speed and memory usage, set the `--preset` argument to one of four available options:
1. exact: maximising accuracy
2. speed: maximising inference speed
3. memory: reducing memory
4. memory_extreme: minimising memory

You can also control the style of the generated text through the `--mode` flag, one of "chat", "factual" or "creative":

```sh
local-gemma-2 --model 9b --preset memory --mode factual
```

To see all available decoding options, call `local-gemma-2 -h`.

## Python Usage

Local Gemma-2 can be run locally through a Python interpreter using the familiar Transformers API. To enable a preset,
import the model class from `local_gemma_2` and pass the `preset` argument to `from_pretrained`. For example, the
following code-snippet loads the [Gemma-2 9b](https://huggingface.co/google/gemma-2-9b) model with the "memory" preset:

```python
from local_gemma_2 import LocalGemma2ForCausalLM
from transformers import AutoTokenizer

model = LocalGemma2ForCausalLM.from_pretrained("google/gemma-2-9b", preset="memory")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

model_inputs = tokenizer("The cat sat on the mat", return_attention_mask=True, return_tensors="pt")
generated_ids = model.generate(**model_inputs.to(model.device))

decoded_text = tokenizer.batch_decode(generated_ids)
```

When using an instruction-tuned model (prefixed by `-it`) for conversational use, prepare the inputs using a
chat-template. The following examples loads [Gemma-2 27b it](https://huggingface.co/google/gemma-2-27b-it) model
using the "auto" preset, which automatically determines the best preset for the device:

```python
from local_gemma_2 import LocalGemma2ForCausalLM
from transformers import AutoTokenizer

model = LocalGemma2ForCausalLM.from_pretrained("google/gemma-2-27b-it", preset="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")

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

## Presets

Local Gemma-2 provides four presets that trade-off accuracy, speed and memory. The following results highlight this
trade-off using [Gemma-2 9b](https://huggingface.co/google/gemma-2-9b) with batch size 1 on an 80GB A100 GPU:

| Mode           | Performance (?) | Inference Speed (tok/s) | Memory (GB) |
|----------------|-----------------|-------------------------|-------------|
| exact          | **a**           | 17.2                    | 18.3        |
| speed          | b               | **18.3**                | 18.3        |
| memory         | c               | 13.8                    | 7.3         |
| memory_extreme | d               | 7.0                     | **4.9**     |


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
| memory         | eager               | int4          | no          |
| memory_extreme | eager               | int2          | yes         |

Note: Due to [Gemma 2 logit softcapping](https://huggingface.co/blog/gemma2#soft-capping-and-attention-implementations), SDPA/FA doesn't work well.
