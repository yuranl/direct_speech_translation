import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
import math, json
import os, glob, codecs

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class asr_translation_model(nn.Module):
    def __init__(self):
        super(asr_translation_model, self).__init__()
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
        self.conv1 = nn.Conv1d(21128, 64, 4, 2,bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 128, 4, 2,bias=True)
        self.conv3 = nn.Conv1d(128, 256, 4, 2,bias=True)
        self.conv4 = nn.Conv1d(256, 512, 4, 2,bias=True)
        self.mask_conv = nn.Conv1d(512, 1, 1, 1,bias=True)
        self.sigmoid = nn.Sigmoid()
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained("./finetuned_model/1213 wd0002/3")

    def forward(self, batch):
        del batch['input_ids']
        del batch['attention_mask']
        logits = self.asr_model(batch['audio_inputs'], attention_mask=batch['audio_inputs_mask']).logits
        translation_input = self.conv4(self.relu(self.conv3(self.relu(self.conv2(self.relu(self.conv1(logits.transpose(1, 2))))))))
        learned_mask = self.sigmoid(self.mask_conv(translation_input))
        translation_input = translation_input * learned_mask
        length = translation_input.shape[2]
        batch_size = translation_input.shape[0]
        new_attention_mask = torch.ones(batch_size,length, device=translation_input.device)
        outputs = self.translation_model(inputs_embeds=translation_input.transpose(1,2), attention_mask = new_attention_mask, labels=batch.labels)
        return outputs

    def generate(self, batch, max_length=512):
        del batch['input_ids']
        del batch['attention_mask']
        logits = self.asr_model(batch['audio_inputs'], attention_mask=batch['audio_inputs_mask']).logits

        translation_input = self.conv4(self.relu(self.conv3(self.relu(self.conv2(self.relu(self.conv1(logits.transpose(1, 2))))))))
        learned_mask = self.sigmoid(self.mask_conv(translation_input))
        translation_input = translation_input * learned_mask
        length = translation_input.shape[2]
        batch_size = translation_input.shape[0]
        new_attention_mask = torch.ones(batch_size,length, device=translation_input.device)
        encoder_output = self.translation_model.get_encoder()(inputs_embeds=translation_input.transpose(1,2),attention_mask = new_attention_mask)
        generated_tokens = self.translation_model.generate(encoder_outputs = encoder_output, attention_mask = new_attention_mask, max_length = max_length)
        return generated_tokens

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
            #print(labels[i])
    return decoded_preds, decoded_labels

def preprocess_function(examples):
    inputs = [ex for ex in examples["transcript"]]
    targets = [ex for ex in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    wav_files = [ex for ex in examples["wav_id"]]
    wav_offset = [ex for ex in examples["offset"]]
    wav_duration = [ex if float(ex) > 0.400 else "0.400" for ex in examples["duration"]]
    wav_files_unique = list(set(wav_files))
    wav_dic = {}
    for i in range(len(wav_files_unique)):
        speech_array, sampling_rate = torchaudio.load('./MT_data/test/audio/' + str(wav_files_unique[i]) + '.wav')
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech = resampler(speech_array.squeeze(0)).numpy()
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
    return model_inputs


model = asr_translation_model()
model.load_state_dict(torch.load('./finetuned_model/asr_translation_final/3'))

for name in tqdm(glob.glob('./MT_data/test/audio/*.wav')):

    json_name = './MT_data/test/' + name.split('\\')[1].split('.')[0] + '.json'
    split_datasets  = load_dataset('json', data_files={'train': './MT_data/test/4.json', 'validation': json_name})

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", return_tensors="pt")
    processor = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")

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
        shuffle=True,
        collate_fn=data_collator,
        batch_size=1,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], 
        collate_fn=data_collator, 
        batch_size=1
    )

    accelerator = Accelerator()
    model, train_dataloader, eval_dataloader = accelerator.prepare(
        model, train_dataloader, eval_dataloader
    )

    model.eval()
    json_file_path = './evaluation/ASR_MT_model_output/' + name.split('\\')[1].split('.')[0] + '_result.json'
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

        f = codecs.open(json_file_path, 'a', "utf-8")
        res = {}
        res['translation'] = decoded_preds[0]
        json.dump(res, f)
        f.write('\n')
        f.close()
