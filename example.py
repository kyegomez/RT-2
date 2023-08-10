import torch 
from rt2.model import RT2

rt2 = RT2()

video = torch.randn(2, 3, 6, 224, 224)
instructions = [
    "bring me an apple on that tree"
]

train_logits = rt2(video, instructions)
rt2.eval()
eval_logits = rt2(video, instructions, cond_scale=2)