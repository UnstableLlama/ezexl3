"""Load the real tensor dependency before legacy optional-dependency stubs.

CPU tensor tests need real Torch even when chat tests mock GPU inference.
Importing Torch does not load a model or initialize CUDA.
"""

import torch  # noqa: F401
