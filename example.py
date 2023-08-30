import torch 
from rt2.model import RT2

model = RT2()

video = torch.randn(2, 3, 6, 224, 224)

instructions = [
    'bring me that apple sitting on the table',
    'please pass the butter'
]

train_logits = model.train(video, instructions)

model.eval()

eval_logits = model.eval(video, instructions, cond_scale=3.)