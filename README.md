# PureDiffusion
A torch implementation for [DDPM](https://github.com/hojonathanho/diffusion) with [DDIM](https://github.com/ermongroup/ddim) sampling. This implementation is not restricted for Image data.

## 1. Installation
`purediffusion` only requires the [`PyTorch`](https://pytorch.org) package. And it is easy to install using [`pip`](https://pypi.org/project/purediffusion/)
```
pip install purediffusion
```

## 2. Tutorial
`purediffusion` provide a simple way to train a diffusion model using customized neural network architectures for arbitrary data dimension.
### 2.1 Pipelines
The script `purediffusion/pipeline.py` provide two diffusion pipelines: `DDPMPipline` and `DDIMPipline`.
#### 2.1.1 DDPM
A DDPMPipline instance prepares all parameters ($\alpha$, $\beta$, $\bar{\alpha}$, etc...) in a specified noisy schedule. To initialize a DDPMPipline:
```
from purediffusion.pipline import DDPMPipline
my_diffusion = DDPMPipline(num_timesteps=1000, device='cpu', beta_schedule='cosine')
```
Now, you can use the instance to sample timesteps and add noise to clean data during training. <br />
Note that a pipline instance is model independent. Meaning: you need to feed model and data dimension to it for generation. For example:
```
my_diffusion.ddpm_reverse(model, batch_size, data_shape)
```
The `data_shape` flag is a list of integers represents the dimension of a single output of the model. e.g., for a $32 \times 32$ RGB image generation model, a possible setting could be `data_shape=[32, 32, 3]`.<br />

#### 2.1.2 DDIM
A `DDPMPipline` instance could be upgraded to a `DDIMPipline` instance for efficient DDIM sampling with specified number of steps: 
```
from purediffusion.pipline import DDPIPipline
my_diffusion_fast = DDIMPipline(my_diffusion, ddim_num_steps=100)
```
To generate data:
```
my_diffusion_fast.ddim_reverse(model, batch_size, data_shape)
```

### 2.2 Model
The neural network instance should take two inputs: 
1. scalar integer timesteps, $t$. The model should convert it to time embedding vectors. <br />
2. noisy data tensor, $D$. The model's output should be of the same shape (dimension) of the noisy data tensor. e.g., $D \in \mathcal{R}^{100 \times 32 \times 32 \times 3}$ for $32 \times 32$ RGB images with batch size $100$.

In the demo, a linear model is provided to demostrate the usage.







