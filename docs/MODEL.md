# Model architectures

| Model | Architecture | Learning Rate | Batch Size | Gradient Steps |
|-------|--------------|---------------|------------|----------------|
| RT-2-PaLI-X-55B | Uses a ViT-22B to process images. The image tokens are passed over a projection layer and then consumed by an encoder-decoder backbone of 32B parameters and 50 layers, similar to UL2. | 1e-3 | 2048 | 80K |
| RT-2-PaLI-X-5B | Uses a ViT-22B to process images. The image tokens are passed over a projection layer and then consumed by an encoder-decoder backbone of 32B parameters and 50 layers, similar to UL2. | 1e-3 | 2048 | 270K |
| RT-2-PaLM-E-12B | Based on a decoder-only LLM that projects robot data such as images and text into the language token space and outputs text such as high-level plans. The visual model used to project images to the language embedding space is a ViT-4B. | 4e-4 | 512 | 1M |
| RT-2-PaLI-3B | Trained on Language-Table, uses a smaller ViT-G/14 (2B parameters) to process images, and UL2-3B for the encoder-decoder network. | 1e-3 | 128 | 300K |