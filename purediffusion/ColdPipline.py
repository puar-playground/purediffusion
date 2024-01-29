import torch
import torch.nn as nn
import numpy as np
import math
from .GaussianPipline import extract, make_beta_schedule
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta):
    # select alphas for computing the variance schedule
    device = alphacums.device
    alphas = alphacums[ddim_timesteps]
    alphas_prev = torch.tensor([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist()).to(device)

    sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    # print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
    # print(f'For the chosen value of eta, which is {eta}, '
    #       f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


class ColdPipline(object):

    def __init__(self, num_timesteps=1000, device='cpu', beta_schedule='cosine'):
        super().__init__()

        self.device = device
        self.num_timesteps = num_timesteps

        self.beta_schedule = beta_schedule
        betas = make_beta_schedule(schedule=beta_schedule, num_timesteps=self.num_timesteps, start=0.0001, end=0.02)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_cumprod)

    def sample_t(self, size=(1,)):
        """Samples batches of time steps to use."""
        t_max = int(self.num_timesteps) - 1
        t = torch.randint(low=0, high=t_max, size=size, device=self.device)
        return t.to(self.device)

    def interpolate_degrade(self, clean_data, data_T, t):

        sqrt_alpha_bar_t = extract(self.alphas_bar_sqrt, t, clean_data)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, clean_data)
        # q(data_t | data_0, data_T)
        noisy_data = sqrt_alpha_bar_t * clean_data + sqrt_one_minus_alpha_bar_t * data_T

        return noisy_data

    def cold_reverse(self, model, data_T, only_last_sample=True):

        model.eval()
        num_timesteps = self.num_timesteps
        data_shape = list(data_T.shape)

        num_t, noisy_data_seq = None, None

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        data_t = data_T.to(self.device)

        pred_data_seq = None
        if only_last_sample:
            num_t = 1
        else:
            pred_data_seq = torch.zeros(data_shape + [num_timesteps]).to(self.device)

        for t in reversed(range(1, num_timesteps)):

            # data_{t-1}
            pred_data_0, data_t = self.taylor_step(model, data_t, data_T, t)

            if only_last_sample:
                num_t += 1
            else:
                pred_data_seq[..., t] = pred_data_0

        if only_last_sample:
            return pred_data_0
        else:
            return pred_data_0, pred_data_seq

    def taylor_step(self, model, data_t, data_T, t):
        t = torch.tensor([t]).to(self.device)
        pred_data_0 = model(data_t, timesteps=t).to(self.device).detach()

        data_t_hat = self.interpolate_degrade(clean_data=pred_data_0, data_T=data_T, t=t)
        data_t_m_1_hat = self.interpolate_degrade(clean_data=pred_data_0, data_T=data_T, t=t - 1)
        data_t_m_1 = data_t - data_t_hat + data_t_m_1_hat

        return pred_data_0, data_t_m_1



class FastColdPipline(ColdPipline):

    def __init__(self, cold_pipline: ColdPipline, ddim_num_steps=50, ddim_discretize="uniform", ddim_eta=0.):
        super().__init__(num_timesteps=cold_pipline.num_timesteps,
                                            device=cold_pipline.device, beta_schedule=cold_pipline.beta_schedule)

        self.ddim_num_steps = ddim_num_steps
        self.ddim_timesteps = self.make_skip_timesteps(ddim_discr_method=ddim_discretize,
                                                       num_ddim_timesteps=ddim_num_steps,
                                                       num_ddpm_timesteps=self.num_timesteps)

    def make_skip_timesteps(self, ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps):
        if ddim_discr_method == 'uniform':
            c = num_ddpm_timesteps // num_ddim_timesteps
            ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
        elif ddim_discr_method == 'new':
            c = (num_ddpm_timesteps - 50) // (num_ddim_timesteps - 50)
            ddim_timesteps = np.asarray(list(range(0, 50)) + list(range(50, num_ddpm_timesteps - 50, c)))
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

        assert ddim_timesteps.shape[0] == num_ddim_timesteps
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        steps_out = ddim_timesteps + 1
        # print(f'Selected timesteps for ddim sampler: {steps_out}')

        return steps_out


if __name__ == "__main__":

    print('Cold diffusion (Interpolation)')

