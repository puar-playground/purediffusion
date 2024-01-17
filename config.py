
class TrainingConfig:
    diffusion_time_steps = 1000
    train_batch_size = 128
    eval_batch_size = 2  # how many images to sample during evaluation
    num_epochs = 500
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    seed = 0

