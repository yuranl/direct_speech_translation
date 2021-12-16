from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
from datasets import load_dataset
import torch
import numpy as np
import glob


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
    return decoded_preds, decoded_labels

def preprocess_function(examples):
    inputs = [ex for ex in examples["transcript"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    return model_inputs

model = AutoModelForSeq2SeqLM.from_pretrained("./finetuned_model/1213 wd0002/3")

for name in glob.glob('./MT_data/test/*.json'):

    split_datasets  = load_dataset('json', data_files={'test': name})
    print(split_datasets)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", return_tensors="pt")

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
        remove_columns=split_datasets["test"].column_names,
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=1,
    )

    accelerator = Accelerator()
    model,test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    model.eval()
    # Test set inference
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=512,
            )

        # Necessary to pad predictions and labels for being gathered
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)

        print(decoded_preds)
        print(decoded_labels)