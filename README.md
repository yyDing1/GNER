<p align="center"><h2 align="center">Rethinking Negative Instances for Generative Named Entity Recognition</h2></p>

<p align="center">
    <a href="https://github.com/yyDing1/GNER/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/yyDing1/GNER"></a>
    <a href="https://huggingface.co/collections/dyyyyyyyy/gner-65dda2cb96c6e35c814dea56"><img alt="Pretrained Models" src="https://img.shields.io/badge/🤗 HuggingFace-Pretrained Models-green"></a>
    <a href="https://arxiv.org/abs/2402.16602"><img alt="Paper" src="https://img.shields.io/badge/📄-Paper-orange">
    <a href="https://opennlg.cn/"><img src="https://img.shields.io/badge/Organization-OpenNLG%20Group-blueviolet"></a>
</p>

We introduce GNER, a **G**enerative **N**amed **E**ntity **R**ecognition framework, which demonstrates enhanced zero-shot capabilities across unseen entity domains. Experiments on two representative generative models, i.e., LLaMA and Flan-T5, show that the integration of negative instances into the training process yields substantial performance enhancements. The resulting models, GNER-T5 and GNER-LLaMA, outperform state-of-the-art (SoTA) approaches by a large margin, achieving improvements of 8 and 11 points in $F_1$ score, respectively. Code and models are publicly available.

* 📖 Paper: [Rethinking Negative Instances for Generative Named Entity Recognition](https://arxiv.org/abs/2402.16602)
* 💾 Models in the 🤗 HuggingFace Hub: [GNER-Models](https://huggingface.co/collections/dyyyyyyyy/gner-65dda2cb96c6e35c814dea56)
* 🔁 Quick Reproduction Materials: [Generation results](model_predictions/)

<p align="center">
<img src="assets/zero_shot_results.png">
</p>

## PreTrained Models

We release five GNER models based on LLaMA (7B) and Flan-T5 (base, large, xl and xxl).

| Model         | # Parameters | Zero-shot Average $F_1$ | Supervised Average $F_1$ |         🤗 HuggingFace <br />Download Link         |
| ------------- | -----------: | :----------------------: | :-----------------------: | :-------------------------------------------------: |
| GNER-LLaMA    |           7B |           66.1           |           86.09           | [link](https://huggingface.co/dyyyyyyyy/GNER-LLaMA-7B) |
| GNER-T5-base  |          77M |           59.5           |           83.21           | [link](https://huggingface.co/dyyyyyyyy/GNER-T5-base) |
| GNER-T5-large |         248M |           63.5           |           85.45           | [link](https://huggingface.co/dyyyyyyyy/GNER-T5-large) |
| GNER-T5-xl    |         783M |           66.1           |           85.94           |  [link](https://huggingface.co/dyyyyyyyy/GNER-T5-xl)  |
| GNER-T5-xxl   |          11B |           69.1           |           86.15           |  [link](https://huggingface.co/dyyyyyyyy/GNER-T5-xxl)  |

## Quick Reproduction

We also provide all the generated results for quick reproduction of our results. The `model_predictions` folder contains the generated results of GNER-LLaMA-7B and GNER-T5-xxl (including the ground truth). You can execute the following commands to evaluate the generated results:

```python
# 0shot performance of GNER-LLaMA
python evaluate.py --tokenizer-path yahma/llama-7b-hf --prediction-path prediction_results/llama-7b-task-adaptation-beam1.jsonl
# 0shot performance of GNER-T5-xxl
python evaluate.py --tokenizer-path google/flan-t5-xxl --prediction-path prediction_results/flan-t5-xxl-task-adaptation-beam1.jsonl
```

Other generated results can be found at [here](https://drive.google.com/drive/folders/1kg7YDRk8jK4_Bo19jJpZtdAQMBoucppW?usp=drive_link), and the execution process is similar to the two examples mentioned above.

## Citation

```bibtex
@misc{ding2024rethinking,
      title={Rethinking Negative Instances for Generative Named Entity Recognition}, 
      author={Yuyang Ding and Juntao Li and Pinzheng Wang and Zecheng Tang and Bowen Yan and Min Zhang},
      year={2024},
      eprint={2402.16602},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
