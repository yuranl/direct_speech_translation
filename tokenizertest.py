from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, get_cosine_schedule_with_warmup
from translationData import translationDataset
from transformers import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
from datasets import load_dataset, load_metric
import torch
import numpy as np
#from sacrebleu.metrics import BLEU

split_datasets  = load_dataset('json', data_files={'train': './transcription_translation/*.json', 'validation': './translation_validation/*.json'})
# translationTrain = translationDataset('./transcription_translation/')
# translationValidation = translationDataset('./translation_validation/')
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
#print(model)
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", return_tensors="pt")

a = tokenizer.encode("额，额，额，额，额，额，额，额，额，额，我是朱永哲")
print(a)
a = [i if i != 3002 else 0 for i in a]
print(a)
print(tokenizer.decode(a))



## 先strip filling words, 再feed into tokenizers?
## find the word dict. Where is it?