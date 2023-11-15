# Temporal-LLMs
Materials for paper "Are large language models temporally grounded?"

![Screentshot](illustration.png)

## Environment Setup

Configure your conda environment from our provided `llama-hf_environment.yml` by,

```
conda env create -f environment.yml
conda activate llama-hf
```

## Prerequisite

### GPT

Our experiments require the inference with these following models,

1. `davinci`,
2. `text-davinci-002`,
3. `text-davinci-003`.

You will need to prepare your own OpenAI API call and access for these models in advance.

### LLaMA Chceckpoints

Our experiments require the inference with these following models,

1. `LLaMA-7B`
2. `LLaMA-13B`
3. `LLaMA-33B`
4. `LLaMA-65B`
5. `LLaMA-2-7B`
6. `LLaMA-2-13B`
7. `LLaMA-2-70B`
8. `LLaMA-2-7B-chat`
9. `LLaMA-2-13B-chat`
10. `LLaMA-2-70B-chat`


We recommend you to download all files from huggingface hub in to a local path, see the LLaMA [here](https://huggingface.co/docs/transformers/main/model_doc/llama) and [here](https://huggingface.co/docs/transformers/main/model_doc/llama2#resources) for LLaMA-2.

## Usage

### McTACO Experiment

#### Inference for GPT & LLaMA

Using the following scripts for testing GPT and LLaMA with `zero/few-shot` prompting,

```
sh run-mctaco-gpt.sh
sh run-mctaco-llama.sh
```

#### Evaluation

We provide our evaluation script based on the original McTACO's evaluation. We recommend you to get familiar with its [original repository](https://github.com/CogComp/MCTACO) as the first step,

```
sh eval-mctaco.sh
```

### Run CaTeRS Experiment

#### Inference for GPT & LLaMA

Using the following scripts for testing GPT and LLaMA models with `few-shot` prompting,

```
sh run-caters-gpt.sh
sh run-caters-llama.sh
```

#### Evaluation

Our evaluation script is strictly following the evaluation of `temporal-bart` model. Again, we recommend you to get familiar with its [repository](https://github.com/jjasonn0717/TemporalBART) as well,

To run the evaluation, simply run this code,
```
python3 eval-caters.py $OUTPUT_PATH $MODEL_NAME
```

Taking `Llama-2-70b-chat-hf` as an example,
```
python3 eval-caters.py llama-output/caters/caters-fs-pt1-output-icl3/ Llama-2-70b-chat-hf
```


### Run TempEval-QA-bi Experiment

#### Inference for GPT & LLaMA

You can use the following scripts for inference with GPT,

```
sh run-tempeval-gpt.sh
```

You can use the following scripts for doing `zero/few-shot + likelihood/decoding-based` evaluation, and `chain-of-thought` experiments for LLaMA models.

```
sh run-tempeval-llama.sh
```

#### Evaluation for Bi-directional Reasoning

To run the bi-directional evaluation in checking model's reasoning consistency, simply run this code,
```
python3 eval-tempeval-bi.py $OUTPUT_PATH $MODEL_NAME
```

Taking `Llama-2-70b-chat-hf` as an example,
```
python3 eval-tempeval-bi.py llama-output/tempeval-qa-bi/fs-bi-pt1-icl3-output-likelihood/ Llama-2-70b-chat-hf
```

#### Evaluation for Chain-of-thought Reasoning
To evaluate the reasoning performance for LLaMA with the chain-of-thought prompting, simply run this code,
```
python3 eval-tempeval-bi-cot.py $OUTPUT_PATH $MODEL_NAME
```

Taking `Llama-2-70b-chat-hf` as an example,
```
python3 eval-tempeval-bi-cot.py llama-output/tempeval-qa-bi/fs-bi-pt1-icl3-cot-output-likelihood/ Llama-2-70b-chat-hf
```

## Model Outputs

We provide all our model's outputs in all datasets in `gpt-output` and `llama-output` for reproducing the results reported in our paper.


## Citation

```
@misc{qiu2023large,
      title={Are Large Language Models Temporally Grounded?}, 
      author={Yifu Qiu and Zheng Zhao and Yftah Ziser and Anna Korhonen and Edoardo M. Ponti and Shay B. Cohen},
      year={2023},
      eprint={2311.08398},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
