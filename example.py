import torch 
from rt2.model import RT2

model = RT2()

video = torch.randn(2, 3, 6, 224, 224)

instructions = [
    'bring me that apple sitting on the table',
    'please pass the butter'
]

train_logits = model(video, instructions) # (2, 6, 11, 256) # (batch, frames, actions, bins)

model.eval()

eval_logits = model(video, instructions, cond_scale = 3.) # classifier free guidance with conditional scale of 3