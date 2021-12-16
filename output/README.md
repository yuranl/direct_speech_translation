# Direct Speech-to-text Translation: Output
This section contains the output and evaluation script for the direct S2T project.
Below are the instructions to run the evaluation script.

## Place the output json files
Place the output `.json` files from the inference scripts into `./output/data`, don't change the folder names.

## Convert to .sgm files
Change the parameters for `json_to_sgm.py` in the python script according to the instructions in the comment (only need to comment and uncomment some variable assignments) according to the type of output being converted.
Run `python ./output/json_to_sgm.py` for each of the output/source/reference types.

## Use Peral script to evalute
Run `perl ./mteval-v13a.pl -r ref{task_id}.sgm -s src{task_id}.sgm -t tst{task_id}.sgm`
and the resulting BLEU scores should be printed to the terminal.