import torch

encoder_ckpt_path = "outputs/tmp/checkpoints/epoch_299-step_6000.ckpt"
ckpt = torch.load(encoder_ckpt_path)
print (ckpt['state_dict'].keys())