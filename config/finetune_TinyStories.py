out_dir = 'out-TinyStories-m3'

eval_interval = 100
eval_iters    = 400
log_interval  = 10

wandb_log = False
wandb_project = 'TinyStories-continue'
wandb_run_name = 'mini-gpt-TinyStories'

dataset = 'TinyStories_new'

init_from = 'resume'

always_save_checkpoint = False

batch_size = 4
block_size = 768
gradient_accumulation_steps = 8

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.05

learning_rate   = 5e-5
max_iters       = 90000
lr_decay_iters  = 90000
min_lr          = 1e-5
beta2           = 0.99
warmup_iters    = 500

compile = False