RT-2 Model Architecture
The RT-2 model architecture is based on the PaLI-X and PaLM-E models, which are vision-language models (VLMs) that process images and text to generate output tokens. The architecture consists of the following components:

Vision Transformer (ViT): This is used to process images. It can accept sequences of n images, leading to n×k tokens per image, where k is the number of patches per image. The image tokens passing over a projection layer is then consumed by an encoder-decoder backbone.

Encoder-Decoder Backbone: This backbone has 32B parameters and 50 layers, similar to UL2. It jointly processes text and images as embeddings to generate output tokens in an auto-regressive manner.

Text Input: The text input usually consists of the type of task and any additional context.

Robot Data Projection: The model projects robot data such as images and text into the language token space and outputs text such as high-level plans. The visual model used to project images to the language embedding space is a ViT-4B.

Concatenation of Continuous Variables: The concatenation of continuous variables to textual input allows the model to be fully multimodal, accepting a wide variety of inputs such as multiple sensor modalities, object-centric representations, scene representations and object entity referrals.

RT-2 Model Pseudocode
The following pseudocode describes the algorithmic process of the RT-2 model:

Initialize Vision Transformer (ViT)
Initialize Encoder-Decoder Backbone
Initialize Text Input
Initialize Robot Data Projection
Initialize Concatenation of Continuous Variables

for each training step do
    Get a batch of images and corresponding text inputs
    Process images through ViT to get image tokens
    Process text inputs to get text tokens
    Pass image and text tokens through Encoder-Decoder Backbone to get output tokens
    Project robot data into language token space to get high-level plans
    Concatenate continuous variables to textual input
    Compute loss between output tokens and ground truth
    Backpropagate loss and update model parameters
end for
RT-2 Model PyTorch Implementation
The following code provides a PyTorch implementation of the RT-2 model:

import torch
import torch.nn as nn
from transformers import ViTModel, BertModel

class RT2Model(nn.Module):
    def __init__(self, num_patches, num_tokens):
        super(RT2Model, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.encoder_decoder = BertModel.from_pretrained('bert-base-uncased')
        self.projection = nn.Linear(self.vit.config.hidden_size, num_patches * num_tokens)
        self.robot_projection = nn.Linear(self.vit.config.hidden_size, num_tokens)

    def forward(self, images, text_inputs, robot_data):
        image_embeddings = self.vit(images).last_hidden_state
        image_tokens = self.projection(image_embeddings)
        encoder_outputs = self.encoder_decoder(inputs_embeds=image_tokens, decoder_input_ids=text_inputs).last_hidden_state
        robot_plan = self.robot_projection(encoder_outputs)
        return robot_plan
Copy code
This code first initializes a Vision Transformer (ViT) and an Encoder-Decoder Backbone (BertModel in this case). In the forward pass, it processes the images through the ViT to get image embeddings, which are then passed through a projection layer to get image tokens. These tokens are then passed through the Encoder-Decoder Backbone along with the text inputs to get the output tokens. Finally, the robot data is projected into the language token space to get the high-level plans.

Please note that this is a simplified version of the model and does not include all the details mentioned in the paper, such as the concatenation of continuous variables to textual input and the specific details of the Encoder-Decoder Backbone. The actual implementation would be more complex and would require a deeper understanding of the models used and the specific details mentioned in the paper.

