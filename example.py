import torch
from rt2.model import RT2

# img: (batch_size, 3, 256, 256)
# caption: (batch_size, 1024)
img = torch.randn(1, 3, 256, 256)
caption = torch.randint(0, 20000, (1, 1024))

# model: RT2
model = RT2()

# Run model on img and caption
output = model(img, caption)
print(output)  # (1, 1024, 20000)
