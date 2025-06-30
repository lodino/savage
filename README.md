# SAVEGE
Codebase for Paper: [Stress-Testing ML Pipelines with Adversarial Data Corruption](https://arxiv.org/pdf/2506.01230).

![image load failed](savage-diagram.png) 

## Important Code Files
- `err_injection.py` contains the code for basic error injection templates, including missing data, selection bias/sampling error, and label error.
- `savage.py` includes the implementation for the TPE (for corruption parameter search) and the beam search (for corruption template search) algorithms.
- `pipelines.py` predefines the pipeline components evaluated in our paper.
- `savage-example.ipynb` shows how to load the predefined pipelines from `pipelines.py` for robustness evaluation, and how to evaluate a customized pipeline for accuracy and fairness metrics.

We also provide a [simple example](https://colab.research.google.com/drive/1j6IiFg-9rJ6nSgpdNoeM_lAsUgH8tV23?usp=sharing) on Google Colab.

