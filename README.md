# Pure-Diffusion
A torch implementation for DDPM with DDIM sampling. The code is not restricted for Image data.

## 1. Installation
Pure-Diffusion only requires the [`PyTorch`](https://pytorch.org) package.
```
conda create --name pure_diffusion python=3.9
conda activate pure_diffusion
python -m pip install torch
```

## 2. Tutorial
Pure-Diffusion provide a simple way to train a diffusion model using customized neural network architectures for arbitrary data dimension.
### 2.1 Pipelines
The script `utils.pipeline.py` provide two diffusion pipelines: `DDPMPipline` and `DDIMPipline`.
#### 2.1.1 DDPM
A DDPMPipline instance will prepare all parameters ($\alpha$, $\beta$, $\bar{\alpha}$) in a specified noisy schedule. To initialize a DDPMPipline:
```
from utils.pipline import DDPMPipline
ddpm_pipeline = DDPMPipline(num_timesteps=1000, device='cpu', beta_schedule='cosine')
```
The Now, you can use the instance to sample timesteps and add noise to clean data during training. <br />
Note that a pipline instance is model independent. Meaning: you need to feed model and data dimension to it for generation. For example:
```
ddpm_pipeline.ddpm_reverse(model, batch_size, data_shape)
```
The `data_shape` flag is a list of integers represents the dimensionality of a single output of the model. e.g., for a $32 \times 32$ image generation model, a possible setting could be `data_shape=[32, 32, 3]`.<br />

#### 2.1.2 DDIM
A `DDPMPipline` instance could be upgraded to a `DDIMPipline` instance for efficient DDIM sampling using specified number of steps: 
```
ddim_pipeline = DDIMPipline(ddpm_pipeline, ddim_num_steps=100)
```
To generate data:
```
ddim_pipeline.ddim_reverse(model, batch_size, data_shape)
```

### 2.2 Model




