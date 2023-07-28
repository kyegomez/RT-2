## Baselines

They compare our method to multiple state-of-the-art baselines that challenge different aspects of our method. All of the baselines use the exact same robotic data.

- **RT-1:** Robotics Transformer 1 (Brohan et al., 2022) is a transformer-based model that achieved state-of-the-art performance on a similar suite of tasks when it was published. The model does not use VLM-based pre-training so it provides an important data point demonstrating whether VLM-based pre-training matters.

- **VC-1:** VC-1 (Majumdar et al., 2023a) is a visual foundation model that uses pre-trained visual representations specifically designed for robotics tasks. They use pre-trained representations from the VC-1 ViT-L model. Since VC-1 does not include language conditioning, They add this by separately embedding the language command via Universal Sentence Encoder (Cer et al., 2018) to enable comparison to our method.

- **R3M:** R3M (Nair et al., 2022b) is a similar method to VC-1 in that R3M uses pre-trained visual-language representations to improve policy training. In this case the authors use Ego4D dataset (Grauman et al., 2022) of human activities to learn the representation that is used by the policy.

- **MOO:** MOO (Stone et al., 2023) is an object-centric approach, where a VLM is first used to specify the object of interest in a form of a single, colored pixel in the original image. This pixel-modified image is then trained with an end-to-end policy to accomplish a set of manipulation tasks.

## Evaluation Details

### Evaluation Scenarios

For studying the emergent capabilities of RT-2 in a quantitative manner, They study various challenging semantic evaluation scenarios that aim to measure capabilities such as reasoning, symbol understanding, and human recognition.

### Evaluation Instructions

Table 2 lists natural language instructions used in model evaluations for unseen objects, backgrounds, and environments. Each instruction was run betTheyen 1-5 times, depending on the number of total instructions in that evaluation set. Table 3 lists natural language instructions used to evaluate quantitative emergent evals. Each instruction was run 5 times.

## Quantitative Experimental Results

### Overall Performance

| Model | Seen Tasks | Unseen Objects | Unseen Backgrounds | Unseen Environments | Unseen Average |
|-------|------------|----------------|--------------------|---------------------|----------------|
| R3M | 45 | 14 | 9 | 0 | 12 |
| VC-1 | 63 | 10 | 3 | 0 | 10 |
| RT-1 | 92 | 43 | 9 | 26 | 32 |
| MOO | 75 | 48 | 41 | 19 | 35 |
| RT-2-PaLI-X-55B (ours) | 91 | 62 | 48 | 63 | 62 |
| RT-2-PaLM-E-12B (ours) | 93 | 76 | 71 | 36 | 62 |

### Emergent Evaluation

| Model | Symbol Understanding | Reasoning | Person Recognition | Average |
|-------|----------------------|-----------|--------------------|---------|
| VC-1 | 11 | 10 | 13 | 11 |
| RT-1 | 16 | 16 | 20 | 17 |
| RT-2-PaLI-X-55B (ours) | 82 | 46 | 53 | 60 |
| RT-2-PaLM-E-12B (ours) | 36 | 43 | 43 | 40 |

### Size and Training Ablations

| Model Size | Training | Unseen Objects | Unseen Backgrounds | Unseen Environments | Average |
|------------|----------|----------------|--------------------|---------------------|---------|
| RT-2-PaLI-X 5B | from scratch | 0 | 0 | 0 | 9 |
| RT-2-PaLI-X 5B | fine-tuning | 24 | 50 | 23 | 42 |
| RT-2-PaLI-X 5B | co-fine-tuning | 60 | 29 | 24 | 44 |
| RT-2-PaLI-X 55B | fine-tuning | 60 | 38 | 19 | 52 |
| RT-2-PaLI-X 55B | co-fine-tuning | 70 | 48 | 35 | 63 |



## Example Failure Cases

In Fig. 9 They provide examples of a notable type of failure case in the Language Table setting, with the RT-2 model not generalizing to unseen object dynamics. In these cases, although the model is able to correctly attend to the language instruction and move to the first correct object, it is not able to control the challenging dynamics of these objects, which are significantly different than the small set of block objects that have been seen in this environment (Lynch et al., 2022).

## Quantitative Experimental Results

### Overall Performance

| Model | Seen Tasks | Unseen Objects | Unseen Backgrounds | Unseen Environments | Unseen Average |
|-------|------------|----------------|--------------------|---------------------|----------------|
| R3M | 45 | 14 | 9 | 0 | 12 |
| VC-1 | 63 | 10 | 3 | 0 | 10 |
| RT-1 | 92 | 43 | 9 | 26 | 32 |
| MOO | 75 | 48 | 41 | 19 | 35 |
| RT-2-PaLI-X-55B (ours) | 91 | 62 | 48 | 63 | 62 |
| RT-2-PaLM-E-12B (ours) | 93 | 76 | 71 | 36 | 62 |

### Emergent Evaluation

| Model | Symbol Understanding | Reasoning | Person Recognition | Average |
|-------|----------------------|-----------|--------------------|---------|
| VC-1 | 11 | 10 | 13 | 11 |
| RT-1 | 16 | 16 | 20 | 17 |
| RT-2-PaLI-X-55B (ours) | 82 | 46 | 53 | 60 |
| RT-2-PaLM-E-12B (ours) | 36 | 43 | 43 | 40 |

### Size and Training Ablations

| Model Size | Training | Unseen Objects | Unseen Backgrounds | Unseen Environments | Average |
|------------|----------|----------------|--------------------|---------------------|---------|
| RT-2-PaLI-X 5B | from scratch | 0 | 0 | 0 | 9 |
| RT-2-PaLI-X 5B | fine-tuning | 24 | 50 | 23 | 42 |
| RT-2-PaLI-X 5B | co-fine-tuning | 60 | 29 | 24 | 44 |
| RT-2-PaLI-X 55B | fine-tuning | 60 | 38 | 19 | 52 |
| RT-2-PaLI-X 55B | co-fine-tuning | 70 | 48 | 35 | 63 |

## Additional Chain-Of-Thought Reasoning Results

They present additional examples of chain-of-thought reasoning rollouts accomplished with RT-2-PaLM-E, as described in Sec. 4.4, in Figure 10.

## Additional Chain-Of-Thought Reasoning Results

They present additional examples of chain-of-thought reasoning rollouts accomplished with RT-2-PaLM-E, as described in Sec. 4.4, in Figure 10.

1. The original pre-training data mixture used in PaLM-E-12B (as described in Driess et al. (2023)) includes robot images for high-level VQA planning tasks that can be similar to images encountered in generalization scenarios. HoTheyver, none of those training examples include low-level actions that are evaluated in this experiment.

## RT-2: Vision-Language-Action Models Transfer Theyb Knowledge to Robotic Control

User: Pick up the object that is different from all other objects
Plan: Pick rxbar chocolate

User: Move the green objects together
Plan: Move green can near green rice chip bag

User: Move the blue objects together
Plan: Move blue chip bag near pepsi

User: I am sleepy, bring me a drink to help.
Plan: Pick redbull can

Figure 10 | Additional examples of RT-2 with chain-of-thought reasoning