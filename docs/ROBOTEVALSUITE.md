# Robot Evaluation Suite


| Evaluation | Description | Instructions | Benchmark |
|------------|-------------|--------------|-----------|
| Performance on Seen Tasks | Measures how well the model performs on tasks that it has been trained on. | Run the model on the training set and measure its performance. | State-of-the-art performance on the training set. |
| Generalization to Unseen Tasks | Measures how well the model can generalize to tasks that it has not been trained on. | Run the model on a set of tasks that are different from the training set but within the same domain. | State-of-the-art performance on a similar suite of unseen tasks. |
| Robustness to Perturbations | Measures how well the model can handle small changes in the input. | Introduce small perturbations to the input and measure the change in the model's output. | The model's output should not change significantly in response to small perturbations in the input. |
| Efficiency | Measures how efficiently the model can complete tasks. | Measure the time it takes for the model to complete a set of tasks. | The model should be able to complete tasks in a reasonable amount of time. |
| Scalability | Measures how well the model can handle larger and more complex tasks. | Run the model on a set of increasingly complex tasks and measure its performance. | The model's performance should not degrade significantly as the complexity of the tasks increases. |