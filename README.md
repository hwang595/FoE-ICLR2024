## Fusing Models with Complementary Expertise
The implementation for ICLR submission: "Fusing Models with Complementary Expertise".

### Overview
---
Training AI models that generalize across tasks and domains has long been among the open problems driving AI research. The emergence of Foundation Models made it easier to obtain expert models for a given task, but the heterogeneity of data that may be encountered at test time often means that any single expert is insufficient. We consider the Fusion of Experts (FoE) problem of fusing outputs of expert models with complementary knowledge of the data distribution and formulate it as an instance of supervised learning. Our method is applicable to both discriminative and generative tasks and leads to significant performance improvements in image and text classification, text summarization, multiple-choice QA, and automatic evaluation of generated text. We also extend our method to the "frugal" setting where it is desired to reduce the number of expert model evaluations at test time.

### Basic Dependencies
* torch, torchvision
* transformer
* datasets
* peft
* scipy, numpy, sklearn, pandas

### Executing experiments
#### An Example Experiment
To run the cifar experiment
```
cd script
bash run_cifar_foe.sh
```

To run the language model experiments
```
cd lm_experiments
python sentiment_analysis_model_embedding.py
```

```
cd lm_experiments
python summarization_model_embedding.py
```

```
cd lm_experiments
bash run_mmlu.sh
```
