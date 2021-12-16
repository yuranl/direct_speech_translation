# Milestone3 explanation
The main python script to run is `finetune.py`

## Input data
Place the `.json` training files in `./transcription_translation/`, see example `4.json`.
Place the validation files `./translation_validation/`, see example `102371.json`.
The script automatically initializes the model with pre-trained parameters from the internet.
Then run `finetune.py`.

## Output
The python terminal will print out the training and validation BLEU scores, as well as training and validation losses
achieved by the model during training.
The trained parameters will be saved in `.\saved_model` directory.