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
import json

#from sacrebleu.metrics import BLEU

split_datasets  = load_dataset('json', data_files={'train': './transcription_translation/*.json', 'validation': './translation_validation/*.json'})

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
# k = model.model.encoder({"size":1})
# print(k)

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", return_tensors="pt")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
translation = pipeline("translation_chinese_to_english", model=model, tokenizer=tokenizer)
text = "我不知道我在干嘛, 这句话是用来测试的"
translated_text = translation(text, max_length=40)[0]['translation_text']
print(translated_text)
max_input_length = 256
max_target_length = 256


# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)


# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook

# model.fc2.register_forward_hook(get_activation('fc2'))
# x = torch.randn(1, 25)
# output = model(x)
# print(activation['fc2'])