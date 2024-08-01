<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local-gemma-2/local_gemma.png?raw=true" width="600"/>
</p>

<h3 align="center">
    <p>Run Gemma-2 locally in Python, fast!</p>
</h3>

This repository provides an easy way to run [Gemma-2](https://huggingface.co/blog/gemma2) locally directly from your CLI (or via a Python library) and fast. It is built on top of the [🤗 Transformers](https://github.com/huggingface/transformers) and [bitsandbytes](https://huggingface.co/docs/bitsandbytes/index) libraries.

It can be configured to give fully equivalent results to the original implementation, or reduce memory requirements down
to just the largest layer in the model!

> [!IMPORTANT]
> There is a new "speed" preset for running `local-gemma` on CUDA ⚡️ It makes use of torch compile for up to 6x faster generation.
> Set `--preset="speed"` when using the CLI, or pass `preset="speed"` to `from_pretrained` when using the Python API

## Installation

There are two installation flavors of `local-gemma`, which you can select depending on your use case:

<details>
  <summary><b><font size="+0.5"><code>pipx</code> - Ideal for CLI</font></b></summary>

  First, follow the installation steps [here](https://github.com/pypa/pipx?tab=readme-ov-file#install-pipx) to install `pipx` on your environment.

  Then, run one of the commands below, depending on your machine.

  #### CUDA

  ```sh
  pipx install local-gemma"[cuda]"
  ```

  #### MPS

  ```sh
  pipx install local-gemma"[mps]"
  ```

  #### CPU

  ```sh
  pipx install local-gemma"[cpu]"
  ```

</details>

<details>
  <summary><b><font size="+0.5"><code>pip</code> - Ideal for Python (CLI + API)</font></b></summary>

  Local Gemma-2 can be installed as a hardware-specific Python package through `pip`. The only requirement is a Python
installation, details for which can be found [here](https://wiki.python.org/moin/BeginnersGuide/Download). You can
check you have a Python installed locally by running:

```sh
python3 --version
```

#### (optional) Create a new Python environment

```sh
python3 -m venv gemma-venv
source gemma-venv/bin/activate
```

#### CUDA

```sh
pip install local-gemma"[cuda]"
```

#### MPS

```sh
pip install local-gemma"[mps]"
```

#### CPU

```sh
pip install local-gemma"[cpu]"
```

</details>

<!---
<details>
  <summary><b><font size="+0.5"><code>Docker</code> - Pre-prepared container</font></b></summary>

  > TODO(SG): add installation

</details>
--->

## CLI Usage

You can chat with the Gemma-2 through an interactive session by calling:

```sh
local-gemma
```

> [!TIP]
> Local Gemma will check for a Hugging Face "read" token to download the model. You can follow [this guide](https://huggingface.co/docs/hub/en/security-tokens) to create a token, and pass it when prompted to log-in. If you're new to Hugging Face and never used a Gemma model, you'll also need to accept the terms at the top of [this page](https://huggingface.co/google/gemma-2-9b-it).

Alternatively, you can request a single output by passing a prompt, such as:

```sh
local-gemma "What is the capital of France?"
```

By default, this loads the [Gemma-2 9b it](https://huggingface.co/google/gemma-2-9b-it) model. To load the [2b it](https://huggingface.co/google/gemma-2-2b-it) or [27b it](https://huggingface.co/google/gemma-2-27b-it)
models, you can set the `--model` argument accordingly:

```sh
local-gemma --model 2b
```

Local Gemma-2 will automatically find the most performant preset for your hardware, trading-off speed and memory. For more
control over generation speed and memory usage, set the `--preset` argument to one of four available options:
1. exact: match the original results by maximizing accuracy
2. speed: maximize throughput through torch compile (CUDA only!)
3. memory: reducing memory through 4-bit quantization
4. memory_extreme: minimizing memory through 4-bit quantization and CPU offload

You can also control the style of the generated text through the `--mode` flag, one of "chat", "factual" or "creative":

```sh
local-gemma --model 9b --preset memory --mode factual
```

Finally, you can also pipe in other commands, which will be appended to the prompt after a `\n` separator

```sh
ls -la | local-gemma "Describe my files"
```

To see all available decoding options, call `local-gemma -h`.

## Python Usage

  > [!NOTE]
  > The `pipx` installation method creates its own Python environment, so you will need to use the `pip` installation method to use this library in a Python script.

Local Gemma-2 can be run locally through a Python interpreter using the familiar Transformers API. To enable a preset,
import the model class from `local_gemma` and pass the `preset` argument to `from_pretrained`. For example, the
following code-snippet loads the [Gemma-2 9b](https://huggingface.co/google/gemma-2-9b) model with the "memory" preset:

```python
from local_gemma import LocalGemma2ForCausalLM
from transformers import AutoTokenizer

model = LocalGemma2ForCausalLM.from_pretrained("google/gemma-2-9b", preset="memory")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

model_inputs = tokenizer("The cat sat on the mat", return_attention_mask=True, return_tensors="pt")
generated_ids = model.generate(**model_inputs.to(model.device))

decoded_text = tokenizer.batch_decode(generated_ids)
```

When using an instruction-tuned model (prefixed by `-it`) for conversational use, prepare the inputs using a
chat-template. The following example loads [Gemma-2 2b it](https://huggingface.co/google/gemma-2-2b-it) model
using the "auto" preset, which automatically determines the best preset for the device:

```python
from local_gemma import LocalGemma2ForCausalLM
from transformers import AutoTokenizer

model = LocalGemma2ForCausalLM.from_pretrained("google/gemma-2-2b-it", preset="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

generated_ids = model.generate(**model_inputs.to(model.device), max_new_tokens=1024, do_sample=True)
decoded_text = tokenizer.batch_decode(generated_ids)
```

## Presets

Local Gemma-2 provides three presets that trade-off accuracy, speed and memory. The following results highlight this
trade-off using [Gemma-2 9b](https://huggingface.co/google/gemma-2-9b) with batch size 1 on an 80GB A100 GPU:

| Mode              | Performance* | Inference Speed (tok/s) | Memory (GB) |
|-------------------|--------------|-------------------------|-------------|
| exact             | **73.0**     | 17.2                    | 18.3        |
| speed (CUDA-only) | **73.0**     | **62.0**                | 19.0        |
| memory            | 72.1         | 13.8                    | **7.3**     |
| memory_extreme    | 72.1         | 13.8                    | **7.3**     |

While an 80GB A100 places the full model on the device, only 3.7GB is required with the `memory_extreme` preset. See the
section [Preset Details](#preset-details) for details.

___
*Zero-shot results averaged over Wino, ARC Easy, Arc Challenge, PIQA, HellaSwag, MMLU, OpenBook QA.

### Preset Details

| Mode              | 2b Min Memory (GB) | 9b Min Memory (GB) | 27b Min Memory (GB) | Weights dtype | CPU Offload |
|-------------------|--------------------|--------------------|---------------------|---------------|-------------|
| exact             | 5.3                | 18.3               | 54.6                | bf16          | no          |
| speed (CUDA-only) | 5.4                | 19.0               | 55.8                | bf16          | no          |
| memory            | 3.7                | 7.3                | 17.0                | int4          | no          |
| memory_extreme    | 1,8                | 3.7                | 4.7                 | int4          | yes         |

`memory_extreme` implements [CPU offloading](https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling) through
[🤗 Accelerate](https://huggingface.co/docs/accelerate/en/index), reducing memory requirements down to the largest layer
in the model (which in this case is the LM head).


## Acknowledgements
Local Gemma-2 is a convenient wrapper around several open-source projects, which we thank explicitly below:
* [Transformers](https://huggingface.co/docs/transformers/en/index) for the PyTorch Gemma-2 implementation. Particularly [Arthur Zucker](https://github.com/ArthurZucker) for adding the model and the logit soft-capping fixes.
* [bitsandbytes](https://huggingface.co/docs/bitsandbytes/index) for the 4-bit optimization on CUDA.
* [quanto](https://github.com/huggingface/optimum-quanto) for the 4-bit optimization on MPS + CPU.
* [Accelerate](https://huggingface.co/docs/accelerate/en/index) for the large model loading utilities.

And last but not least, thank you to Google for the pre-trained [Gemma-2 checkpoints](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315), all of which you can find on the Hugging Face Hub.
