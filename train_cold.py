import torch
import torch.nn.functional as F
from purediffusion.optimization import get_cosine_schedule_with_warmup
from purediffusion.dataset import syth_data
import torch.utils.data as data
from purediffusion.model import LinearDiffusion
from config import TrainingConfig
from tqdm import tqdm
from purediffusion.cold import Coldpipline
import time

def glorot_initialize(data_shape):
    init_data = torch.zeros(data_shape)
    gain = torch.nn.init.calculate_gain('relu')
    torch.nn.init.xavier_normal_(init_data, gain=gain)
    return init_data


def train_loop(config, model, cold_pipline, optimizer, train_data):

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

                # Sample a random timestep for each image
                bs = clean_data.shape[0]
                timesteps = cold_pipline.sample_t(size=(bs,))

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                data_T = glorot_initialize(clean_data.shape)
                noisy_data = cold_pipline.interpolate_degrade(clean_data, data_T, timesteps)

                # Predict the noise residual
                data_pred = model(noisy_data, timesteps)
                cold_diff_loss = F.mse_loss(data_pred, clean_data)
                constraint_loss = F.mse_loss(torch.sum(data_pred, dim=-1), torch.ones([bs]))
                loss = cold_diff_loss + constraint_loss

                # update parameters
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # update pbar
                progress_bar.update(1)
                logs = {"cold_diff": cold_diff_loss.detach().item(), "constraint": constraint_loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                global_step += 1

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        test_data_shape = list(clean_data.shape)
        test_data_shape[0] = 2
        data_T_test = glorot_initialize(test_data_shape)
        generated = cold_pipline.cold_reverse(model, data_T_test)
        print('>' * 150)
        print('generated')
        print(generated)
        print('center')
        print(train_data.center)
        time.sleep(0.1)
        states = model.state_dict()
        torch.save(states, './checkpoint/cold_test.pt')


if __name__ == "__main__":

    config = TrainingConfig()
    y_dim = 5
    train_data = syth_data(50000, y_dim)
    model = LinearDiffusion(n_steps=config.diffusion_time_steps, y_dim=y_dim, feature_dim=512)
    cold_pipline = Coldpipline(num_timesteps=100, device='cpu', beta_schedule='cosine')

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_loop(config, model, cold_pipline, optimizer, train_data)


