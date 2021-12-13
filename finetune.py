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

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
#print(model)
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", return_tensors="pt")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
translation = pipeline("translation_chinese_to_english", model=model, tokenizer=tokenizer)
text = "我不知道我在干嘛, 这句话是用来测试的"
translated_text = translation(text, max_length=40)[0]['translation_text']
print(translated_text)
max_input_length = 256
max_target_length = 256


def preprocess_function(examples):
    inputs = [ex for ex in examples["transcript"]]
    targets = [ex for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=16,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=16
)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.0001)
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
metric = load_metric("sacrebleu")
#metric = BLEU

num_train_epochs = 20
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
wu_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_update_steps_per_epoch * 2,
    num_training_steps=num_training_steps,
    num_cycles=6,
)
cosine_scheduler = get_cosine_schedule_with_warmup(
    optimizer = optimizer,
    num_warmup_steps=num_update_steps_per_epoch * 2,
    num_training_steps=num_training_steps,
    num_cycles=6
)
# print(lr_scheduler)
# print(wu_scheduler)

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

def pretrained_performance():
    model.eval()
    loss_eval_cumu = 0
    i = 0
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            i += 1
            loss_eval_cumu += loss.item()
        
    loss_eval_cumu /= i
    print(f"Before training, Validation Loss: {loss_eval_cumu:.2f}")
        
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
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    results = metric.compute()
    print(f"Before training, BLEU score: {results['score']:.2f}")


pretrained_performance()
progress_bar = tqdm(range(num_training_steps))
loss_train = []
loss_eval = []
for epoch in range(num_train_epochs):
    # Training
    model.train()
    loss_cumu = 0
    
    i = 0
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        # lr_scheduler.step()
        # wu_scheduler.step()
        cosine_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        i += 1
        loss_cumu += loss.item()
    
    loss_cumu /= i
    loss_train.append(loss_cumu)
    print(f"epoch {epoch + 1}, Training Loss: {loss_cumu:.2f}")

    # Evaluation
    model.eval()
    loss_eval_cumu = 0
    i = 0
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            i += 1
            loss_eval_cumu += loss.item()
        
    loss_eval_cumu /= i
    loss_eval.append(loss_eval_cumu)
    print(f"epoch {epoch + 1}, Validation Loss: {loss_eval_cumu:.2f}")

    if epoch % 5 == 0:
        model.eval()

        # Training BLEU
        train_batch_count = 0
        for batch in tqdm(train_dataloader):
            train_batch_count += 1
            if (train_batch_count > 262): break
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

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        
        results = metric.compute()
        print(f"epoch {epoch + 1}, Training BLEU score: {results['score']:.2f}")

        # Validation BLEU
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
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        results = metric.compute()
        print(f"epoch {epoch + 1}, Validation BLEU score: {results['score']:.2f}")


    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    output_dir = "finetuned_model/" + str(epoch + 1)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

print(loss_train)
# [2.282253554751796, 2.05841838108973, 1.9250032483287283, 1.8195626936160183, 1.733216897190409, 1.6580969759985889, 1.591225100903855, 1.5336571980755516, 1.4812494712604292,1.435242395784416,
# 1.3950268624516546, 1.3606918328061728, 1.3294000998563513, 1.3001842817726352, 1.2769479078914001, 1.2572869315206212, 1.2393105868709609, 1.2250999408496626, 1.2153578630774027, 1.2093690811343618]
print(loss_eval)
# [2.282338900420502, 2.267384870361736, 2.264929485002547, 2.2666508701011425, 2.2798181569758262, 2.2922261036534346, 2.306174605174829, 2.317114326107593, 2.328502018033093, 2.3420375635605732,
# 2.3550392694145668, 2.365725151455129, 2.3725518775350265, 2.386785459427433, 2.393552503513016, 2.3997718673625976, 2.4036135395974605, 2.410205786009781, 2.4129696329131383, 2.413819869056003]

# BLEU: 0 - 12.39, 1 - 15.50, 6 - 15.73, 11 - 15.63, 16 - 15.42