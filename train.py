import torch
import torch.nn.functional as F
from utils.optimization import get_cosine_schedule_with_warmup
from utils.dataset import syth_data
import torch.utils.data as data
from utils.model import LinearDiffusion
from config import TrainingConfig
from tqdm import tqdm
from utils.pipline import DDIMPipline, DDPMPipline
import time

def train_loop(config, model, ddim_pipeline, optimizer, train_data):

    train_dataloader = data.DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True, num_workers=4)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        with tqdm(total=len(train_dataloader)) as progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")

            for step, clean_data in enumerate(train_dataloader):

                # Sample noise to add to the images
                noise = torch.rand_like(clean_data)
                bs = clean_data.shape[0]

                # Sample a random timestep for each image
                timesteps = ddim_pipeline.sample_t(size=(bs,))

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_data = ddim_pipeline.add_noise(clean_data, noise, timesteps)

                # Predict the noise residual
                noise_pred = model(noisy_data, timesteps)
                loss = F.mse_loss(noise_pred, noise)

                # update parameters
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # update pbar
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                global_step += 1


        generated = ddim_pipeline.ddim_reverse(model, batch_size=2, data_shape=[y_dim])
        print('>' * 150)
        print('generated')
        print(generated)
        print('center')
        print(train_data.center)
        time.sleep(0.1)
        states = model.state_dict()
        torch.save(states, './model/test.pt')


if __name__ == "__main__":

    config = TrainingConfig()
    y_dim = 5
    train_data = syth_data(100000, y_dim)
    model = LinearDiffusion(n_steps=config.diffusion_time_steps, y_dim=y_dim, feature_dim=512)
    ddpm_pipeline = DDPMPipline(num_timesteps=1000, device='cpu', beta_schedule='cosine')
    ddim_pipeline = DDIMPipline(ddpm_pipeline, ddim_num_steps=100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_loop(config, model, ddim_pipeline, optimizer, train_data)


