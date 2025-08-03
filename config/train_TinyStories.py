out_dir = 'out-TinyStories-m3'

eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = True

wandb_log = False
wandb_project = 'TinyStories'
wandb_run_name = 'mini-gpt-TinyStories'

dataset = 'TinyStories_new'

batch_size = 4
block_size = 768
gradient_accumulation_steps = 8

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.05

learning_rate = 1e-3
max_iters = 80000
lr_decay_iters = 80000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 2000

compile = False