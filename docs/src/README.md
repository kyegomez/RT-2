[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Robotic Transformer 2 (RT-2): The Vision-Language-Action Model
![rt gif](rt.gif)

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/RT-2)](https://github.com/kyegomez/RT-2/issues) 
[![GitHub forks](https://img.shields.io/github/forks/kyegomez/RT-2)](https://github.com/kyegomez/RT-2/network) 
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/RT-2)](https://github.com/kyegomez/RT-2/stargazers) 
[![GitHub license](https://img.shields.io/github/license/kyegomez/RT-2)](https://github.com/kyegomez/RT-2/blob/master/LICENSE)
[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/RT-2)](https://twitter.com/intent/tweet?text=Excited%20to%20introduce%20RT-2,%20the%20all-new%20robotics%20model%20with%20the%20potential%20to%20revolutionize%20automation.%20Join%20us%20on%20this%20journey%20towards%20a%20smarter%20future.%20%23RT1%20%23Robotics&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FRT-2)
[![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FRT-2)[![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FRT-2&title=Introducing%20RT-2%2C%20the%20All-New%20Robotics%20Model&summary=RT-2%20is%20the%20next-generation%20robotics%20model%20that%20promises%20to%20transform%20industries%20with%20its%20intelligence%20and%20efficiency.%20Join%20us%20to%20be%20a%20part%20of%20this%20revolutionary%20journey%20%23RT1%20%23Robotics&source=)
![Discord](https://img.shields.io/discord/999382051935506503)
[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FRT-2&title=Exciting%20Times%20Ahead%20with%20RT-2%2C%20the%20All-New%20Robotics%20Model%20%23RT1%20%23Robotics)
[![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FRT-2&t=Exciting%20Times%20Ahead%20with%20RT-2%2C%20the%20All-New%20Robotics%20Model%20%23RT1%20%23Robotics)
[![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FRT-2&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=RT-2%2C%20the%20Revolutionary%20Robotics%20Model%20that%20will%20Change%20the%20Way%20We%20Work%20%23RT1%20%23Robotics)
[![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=I%20just%20discovered%20RT-2,%20the%20all-new%20robotics%20model%20that%20promises%20to%20revolutionize%20automation.%20Join%20me%20on%20this%20exciting%20journey%20towards%20a%20smarter%20future.%20%23RT1%20%23Robotics%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FRT-2)

</div>

---


This is my implementation of the model behind RT-2. RT-2 leverages PALM-E as the backbone with a Vision encoder and language backbone where images are embedded and concatenated in the same space as the language embeddings. This architecture is quite easy to architect but suffers from a lack of deep understanding of both the unified multi modal representation or the individual modality representations.

[CLICK HERE FOR THE PAPER](https://robotics-transformer2.github.io/assets/rt2.pdf)


## Installation

RT-2 can be easily installed using pip:

```bash
pip install rt2
```
# Usage


The `RT2` class is a PyTorch module that integrates the PALM-E model into the RT-2 class. Here are some examples of how to use it:

#### Initialization

First, you need to initialize the `RT2` class. You can do this by providing the necessary parameters to the constructor:

```python

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


```


## Benefits

RT-2 stands at the intersection of vision, language, and action, delivering unmatched capabilities and significant benefits for the world of robotics.

- Leveraging web-scale datasets and firsthand robotic data, RT-2 provides exceptional performance in understanding and translating visual and semantic cues into robotic control actions.
- RT-2's architecture is based on well-established models, offering a high chance of success in diverse applications.
- With clear installation instructions and well-documented examples, you can integrate RT-2 into your systems quickly.
- RT-2 simplifies the complexities of multi-domaster understanding, reducing the burden on your data processing and action prediction pipeline.

## Model Architecture

RT-2 integrates a high-capacity Vision-Language model (VLM), initially pre-trained on web-scale data, with robotics data from RT-2. The VLM uses images as input to generate a sequence of tokens representing natural language text. To adapt this for robotic control, RT-2 outputs actions represented as tokens in the model’s output.

RT-2 is fine-tuned using both web and robotics data. The resultant model interprets robot camera images and predicts direct actions for the robot to execute. In essence, it converts visual and language patterns into action-oriented instructions, a remarkable feat in the field of robotic control.


## Datasets
Datasets used in the paper


| Dataset | Description | Source | Percentage in Training Mixture (RT-2-PaLI-X) | Percentage in Training Mixture (RT-2-PaLM-E) |
|---------|-------------|--------|----------------------------------------------|----------------------------------------------|
| WebLI | Around 10B image-text pairs across 109 languages, filtered to the top 10% scoring cross-modal similarity examples to give 1B training examples. | Chen et al. (2023b), Driess et al. (2023) | N/A | N/A |
| Episodic WebLI | Not used in co-fine-tuning RT-2-PaLI-X. | Chen et al. (2023a) | N/A | N/A |
| Robotics Dataset | Demonstration episodes collected with a mobile manipulation robot. Each demonstration is annotated with a natural language instruction from one of seven skills. | Brohan et al. (2022) | 50% | 66% |
| Language-Table | Used for training on several prediction tasks. | Lynch et al. (2022) | N/A | N/A |




## Commercial Use Cases

The unique capabilities of RT-2 open up numerous commercial applications:

- **Automated Factories**: RT-2 can significantly enhance automation in factories by understanding and responding to complex visual and language cues.
- **Healthcare**: In robotic surgeries or patient care, RT-2 can assist in understanding and performing tasks based on both visual and verbal instructions.
- **Smart Homes**: Integration of RT-2 in smart home systems can lead to improved automation, understanding homeowner instructions in a much more nuanced manner.


## Contributing

Contributions to RT-2 are always welcome! Feel free to open an issue or pull request on the GitHub repository.

## Contact

For any queries or issues, kindly open a GitHub issue or get in touch with [kyegomez](https://github.com/kyegomez).

## Citation

```bibtex
@inproceedings{RT-2,2023,
  title={},
  author={Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski,
Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu,
Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog,
Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang,
Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch,
Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi,
Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong,
Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu,
and Brianna Zitkovich},
  year={2024}
}
```


## License

RT-2 is provided under the MIT License. See the LICENSE file for details.