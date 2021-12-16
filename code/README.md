# Direct Speech-to-text Translation: Code
This section contains the code for the direct S2T project.
Below are the instructions to run these code.

## Installation
A python version 3.8 environment is needed.
See `requirements.txt` for python modules used; use `pip install requirements.txt` to install required modules.

## Data
To run the scripts correctly, ensure that data is found at `./data/` relative to the root.

## Simple baseline
See `simple_baseline.py` for the script used in the simple baseline (a cascade of off-the-shelf models).
At the root of the project, run `python ./code/simple_baseline.py` to run this script.
The result of translation will be saved to `./evaluation/baseline_model_output/` in `.json` format, including translations of the `.wav` inputs only.

## Strong baseline: MT model
Our strong baseline can be found in `strong_baseline.py`. 
At the root of the project, run `python ./code/strong_baseline.py` to run this script.
The result of the translation will be saved to `./code/test/` folder in `.json` format.

## Finetuning of MT model
Our training script for the MT model can be found in `MT_finetune.py`.
Our inference script for the MT model can be found in `MT_inference.py`.
Run these scripts at the root of the project.
If running the inference script, the result of the translation will be saved to `./evaluation/MT_model_output/` in `.json` format.
Also, see `MT_finetune-tokenizertest.py` for our attempt at training with a different tokenizer.

## ASR model
See `asr.py` for a baseline pretrained ASR model.

## S2T model
For final speech-to-text translation task,
the training script can be found in `S2T.py`;
the inference script can be found in `S2T_inference.py`.
Run these scripts at the root of the project: `python ./code/S2T_inference.py`.
If running the inference script, the result of the translation will be saved to `./evaluation/ASR_MT_model_output/` in `.json` format.