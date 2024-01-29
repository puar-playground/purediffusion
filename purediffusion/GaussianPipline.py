import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

def extract(input, t, x):
    device = input.device
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape).to(device)

def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(num_timesteps)])
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    return betas


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


class DDPMPipline(object):

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
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.device), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coeff_2 = (torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = posterior_variance
        self.logvar = betas.log()


    def sample_t(self, size=(1,)):
       """Samples batches of time steps to use."""
       t_max = int(self.num_timesteps) - 1
       t = torch.randint(low=0, high=t_max, size=size, device=self.device)

       return t.to(self.device)

    def add_noise(self, clean_data, noise, t):

        sqrt_alpha_bar_t = extract(self.alphas_bar_sqrt, t, clean_data)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, clean_data)
        # q(data_t | data_0, x)
        noisy_data = sqrt_alpha_bar_t * clean_data + sqrt_one_minus_alpha_bar_t * noise

        return noisy_data

    def reparam(self, noisy_data, predicted_noise, t):
        device = noisy_data.device
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, noisy_data)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        reparam_data = 1 / sqrt_alpha_bar_t * (noisy_data - predicted_noise * sqrt_one_minus_alpha_bar_t).to(device)

        return reparam_data

    def ddpm_reverse(self, model, batch_size, data_shape, only_last_sample=True, stochastic=True):

        model.eval()
        data_shape = [batch_size] + data_shape
        num_timesteps = self.num_timesteps

        num_t, noisy_data_seq = None, None
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # middle steps
        # data_T: noisy_data
        noisy_data = stochastic * torch.randn_like(torch.zeros(data_shape)).to(self.device)

        if only_last_sample:
            num_t = 1
        else:
            noisy_data_seq = torch.zeros(data_shape + [num_timesteps + 1]).to(self.device)
            noisy_data_seq[..., num_timesteps] = noisy_data

        for t in reversed(range(1, num_timesteps - 1)):

            noisy_data = self.ddpm_step(model, noisy_data, t, stochastic=stochastic)
            if only_last_sample:
                num_t += 1
            else:
                noisy_data_seq[..., t] = noisy_data
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # last step
        if only_last_sample:
            predicted_data = self.ddpm_step_t_1to0(model, noisy_data)
            return predicted_data
        else:
            predicted_data = self.ddpm_step_t_1to0(model, noisy_data_seq[..., 1])
            noisy_data_seq[..., 0] = predicted_data

            return predicted_data, noisy_data_seq

    def ddpm_step(self, model, noisy_data_t, t, stochastic):
        z = stochastic * torch.randn_like(noisy_data_t)
        t = torch.tensor([t]).to(self.device)
        alpha_t = extract(self.alphas, t, noisy_data_t)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, noisy_data_t)
        sqrt_one_minus_alpha_bar_t_m_1 = extract(self.one_minus_alphas_bar_sqrt, t - 1, noisy_data_t)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
        # noisy_data_t_m_1 posterior mean component coefficients
        gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
        gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())

        predicted_noise = model(noisy_data_t, timesteps=t).to(self.device).detach()

        # clean_data reparameterization
        reparam_data = 1 / sqrt_alpha_bar_t * (noisy_data_t - predicted_noise * sqrt_one_minus_alpha_bar_t).to(self.device)

        # posterior mean
        noisy_data_t_m_1_hat = gamma_0 * reparam_data + gamma_1 * noisy_data_t

        # posterior variance
        beta_t_hat = (sqrt_one_minus_alpha_bar_t_m_1.square()) / (sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)
        noisy_data_t_m_1 = noisy_data_t_m_1_hat.to(self.device) + beta_t_hat.sqrt().to(self.device) * z.to(self.device)

        return noisy_data_t_m_1

    def ddpm_step_t_1to0(self, model, noisy_data):
        device = next(model.parameters()).device
        t = torch.tensor([0]).to(device)  # corresponding to timestep 1 (i.e., t=1 in diffusion models)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, noisy_data)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        predicted_noise = model(noisy_data, timesteps=t).to(device).detach()

        # y_0 reparameterization
        reparam_data = 1 / sqrt_alpha_bar_t * (noisy_data - predicted_noise * sqrt_one_minus_alpha_bar_t).to(device)

        return reparam_data.to(device)


class DDIMPipline(DDPMPipline):

    def __init__(self, ddpm_pipeline: DDPMPipline, ddim_num_steps=50, ddim_discretize="uniform", ddim_eta=0.):
        super().__init__(num_timesteps=ddpm_pipeline.num_timesteps,
                                            device=ddpm_pipeline.device, beta_schedule=ddpm_pipeline.beta_schedule)

        self.ddim_num_steps = ddim_num_steps
        self.ddim_timesteps = self.make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps)

        # alphas have to be defined for each timestep
        assert self.alphas_cumprod.shape[0] == self.num_timesteps

        # ddim sampling parameters
        self.ddim_sigmas, self.ddim_alphas, self.ddim_alphas_prev = \
            make_ddim_sampling_parameters(alphacums=self.alphas_cumprod,
                                          ddim_timesteps=self.ddim_timesteps, eta=ddim_eta)
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1. - self.ddim_alphas)

        self.sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))

    def make_ddim_timesteps(self, ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps):
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

    def ddim_reverse(self, model, batch_size, data_shape, only_last_sample=True, stochastic=True):

        model.eval()
        data_shape = [batch_size] + data_shape

        intermediates = None
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        noisy_data = 1 * stochastic * torch.randn_like(torch.zeros(data_shape)).to(self.device)

        if not only_last_sample:
            intermediates = {'data_inter': [noisy_data], 'pred_data': [noisy_data]}

        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)

            noisy_data, reparam_data = self.ddim_step(model, noisy_data, t, index)

            if not only_last_sample:
                intermediates['data_inter'].append(noisy_data)
                intermediates['pred_data'].append(reparam_data)

        if only_last_sample:
            return noisy_data
        else:
            return noisy_data, intermediates

    def ddim_step(self, model, noisy_data, t, index):

        device = next(model.parameters()).device
        e_t = model(noisy_data, timesteps=t).to(device).detach()

        sqrt_one_minus_alphas = torch.sqrt(1. - self.ddim_alphas)
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full(e_t.shape, self.ddim_alphas[index], device=device)
        a_t_m_1 = torch.full(e_t.shape, self.ddim_alphas_prev[index], device=device)
        sigma_t = torch.full(e_t.shape, self.ddim_sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(e_t.shape, sqrt_one_minus_alphas[index], device=device)

        # direction pointing to data_t
        dir_b_t = (1. - a_t_m_1 - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * torch.randn_like(noisy_data).to(device)

        # reparameterize data_0
        reparam_data = (noisy_data - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # compute data_t_m_1
        noisy_data_t_m_1 = a_t_m_1.sqrt() * reparam_data + 1 * dir_b_t + noise

        return noisy_data_t_m_1, reparam_data


if __name__ == "__main__":

    print('Gaussian diffusion code')