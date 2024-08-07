# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools


DEPS = [
    "setuptools",
    "torch>=2.1.1",
    "accelerate>=0.33.0",
    "transformers>=4.44.0",
]

EXTRA_CUDA_DEPS = ["bitsandbytes>=0.43.2"]
EXTRA_MPS_DEPS = ["quanto>=0.2.0", "torch>=2.4.0"]

setuptools.setup(
    name='local_gemma',
    version='0.2.0',
    author="The Hugging Face team",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': ['local-gemma=local_gemma.cli:main']
        },
    install_requires=DEPS,
    extras_require={
        "cuda": EXTRA_CUDA_DEPS,
        "mps": EXTRA_MPS_DEPS,
        "cpu": EXTRA_MPS_DEPS,
    },
)
