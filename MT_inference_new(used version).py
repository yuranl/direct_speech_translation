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
import codecs
import numpy as np
import glob, json

def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    for i in range(len(decoded_labels)):
        if (decoded_labels[i] == ['']):
            decoded_labels[i] = ['This sentence was corrupted. Developer notes']
            print(labels[i])
    return decoded_preds, decoded_labels

def preprocess_function(examples):
    inputs = [ex for ex in examples["transcript"]]
    targets = [ex for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


for name in glob.glob('./MT_data/test/*.json'):

    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    #print(model)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", return_tensors="pt")
    split_datasets  = load_dataset('json', data_files={'train': './transcription_translation/*.json', 'validation': name})


    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    translation = pipeline("translation_chinese_to_english", model=model, tokenizer=tokenizer)
    text = "我不知道我在干嘛, 这句话是用来测试的"
    translated_text = translation(text, max_length=40)[0]['translation_text']
    print(translated_text)
    max_input_length = 256
    max_target_length = 256

    tokenized_datasets = split_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=split_datasets["train"].column_names,
    )
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=1,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], collate_fn=data_collator, batch_size=1
    )

    accelerator = Accelerator()
    model, train_dataloader, eval_dataloader = accelerator.prepare(
        model, train_dataloader, eval_dataloader
    )
    metric = load_metric("sacrebleu")

    # Validation BLEU

    json_file_path = './evaluation-1214/MT_model_output/' + name.split('\\')[1].split('.')[0] + '_result.json'
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=512,
            )
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)

        f = codecs.open(json_file_path, 'a', "utf-8")
        json_line = {}
        json_line["translation"] = decoded_preds[0]
        json_line = json.dumps(json_line)
        f.write(json_line + '\n')
        f.close()