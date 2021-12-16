import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
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
import torch.nn as nn
import numpy as np
import math
import torch.optim as optim

class asr_translation_model(nn.Module):
    def __init__(self):
        super(asr_translation_model, self).__init__()
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")


    def forward(self, batch):
        del batch['input_ids']
        del batch['attention_mask']
        outputs = self.asr_model(batch['audio_inputs'], attention_mask=batch['audio_inputs_mask']).logits
        # translation_input = self.conv2(self.relu(self.conv1(logits.transpose(1, 2))))
        length = outputs.shape[2]
        batch_size = outputs.shape[0]
        new_attention_mask = torch.ones(batch_size, length, device=outputs.device)
        return outputs

    def generate(self, batch, max_length=512):
        del batch['input_ids']
        del batch['attention_mask']
        outputs = self.asr_model(batch['audio_inputs'], attention_mask=batch['audio_inputs_mask']).logits
        length = outputs.shape[2]
        batch_size = outputs.shape[0]
        new_attention_mask = torch.ones(batch_size,length, device=outputs.device)
        return generated_tokens


split_datasets  = load_dataset('json', data_files={'train': './transcription_translation/6.json', 'validation': './translation_validation/102371.json'})
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")


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
    # model_inputs = {}
    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    wav_files = [ex for ex in examples["wav_id"]]
    wav_offset = [ex for ex in examples["offset"]]
    wav_duration = [ex if float(ex) > 0.400 else "0.400" for ex in examples["duration"]]
    wav_files_unique = list(set(wav_files))
    wav_dic = {}
    for i in range(len(wav_files_unique)):
        speech_array, sampling_rate = torchaudio.load('./data_source/audio_train/' + str(wav_files_unique[i]) + '.wav')
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech = resampler(speech_array.squeeze(0)).numpy()
        print(speech.shape)
        wav_dic[str(wav_files_unique[i])] = speech
    batch_speech_sample = []
    for i in range(len(wav_files)):
        whole_audio = wav_dic[str(wav_files[i])]
        index_start = math.floor(float(wav_offset[i]) * 16000)
        index_end = math.ceil((float(wav_duration[i]) + float(wav_offset[i])) * 16000)
        batch_speech_sample.append(whole_audio[index_start:index_end])
    audio_batch_inputs = processor(batch_speech_sample, sampling_rate=16_000, return_tensors="pt", padding=True)
    model_inputs["audio_inputs"] = audio_batch_inputs.input_values.tolist()
    model_inputs["audio_inputs_mask"] = audio_batch_inputs.attention_mask.tolist()
    # del model_inputs['input_ids']
    # del model_inputs['attention_mask']
    return model_inputs

def pretrained_performance():
    model.eval()
    loss_eval_cumu = 0
    i = 0
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch)
            loss = outputs.loss
            i += 1
            loss_eval_cumu += loss.item()

    loss_eval_cumu /= i
    print(f"Before training, Validation Loss: {loss_eval_cumu:.2f}")

    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch,
                max_length=256,
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




for wdecay in [0.0002]:

    model = asr_translation_model()
    #print(model)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", return_tensors="pt")
    processor = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
    # asr_model = Wav2Vec2ForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    max_input_length = 128
    max_target_length = 128

    tokenized_datasets = split_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=split_datasets["train"].column_names,
    )
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=1,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], 
        collate_fn=data_collator, 
        batch_size=1
    )

    optimizer = optim.SGD(model.parameters(), lr=2e-4)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    metric = load_metric("sacrebleu")

    num_train_epochs = 8
    accum_iter = 16
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accum_iter)
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
    
    progress_bar = tqdm(range(num_training_steps))
    loss_train = []
    loss_eval = []
    for param in model.asr_model.parameters():
        param.requires_grad = False
    for param in model.translation_model.get_decoder().parameters():
        param.requires_grad = False

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        loss_cumu = 0

        i = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            with torch.set_grad_enabled(True):
                outputs = model(batch)
                loss = outputs.loss
                loss = loss / accum_iter
                accelerator.backward(loss)

                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()
                    cosine_scheduler.step()
                    progress_bar.update(1)
                    i += 1
                   
                loss_cumu += float(loss.item())

        loss_cumu /= i
        loss_train.append(loss_cumu)
        print(f"epoch {epoch + 1}, Training Loss: {loss_cumu:.2f}")

        # Evaluation
        model.eval()
        loss_eval_cumu = 0
        i = 0
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                outputs = model(batch)
                loss = outputs.loss
                i += 1
                loss_eval_cumu += float(loss.item())

        loss_eval_cumu /= i
        loss_eval.append(loss_eval_cumu)
        print(f"epoch {epoch + 1}, Validation Loss: {loss_eval_cumu:.2f}")

        if epoch in [0, 1, 2, 3, 4, 7]:
            model.eval()

            # Training BLEU
            train_batch_count = 0
            for batch in tqdm(train_dataloader):
                train_batch_count += 1
                if (train_batch_count > 262): break
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch,
                        max_length=256,
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
                        batch,
                        max_length=256,
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
            print("validation reference:")
            print(decoded_labels[0:20])
            print("model predictions:")
            print(decoded_preds[0:20])
            print(f"epoch {epoch + 1}, Validation BLEU score: {results['score']:.2f}")


            # Save and upload
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            output_dir = "finetuned_model/" + str(epoch + 1)
            torch.save(model.state_dict(), output_dir)
