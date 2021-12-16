import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# test_dataset = load_dataset("common_voice", "zh-CN", split="test")

processor = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
model = Wav2Vec2ForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")



# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

# test_dataset = test_dataset.map(speech_file_to_array_fn)
speech_array, sampling_rate = torchaudio.load('data_source/audio_train/4.wav')
print(sampling_rate)
resampler = torchaudio.transforms.Resample(sampling_rate, 16_000)
speech = resampler(speech_array).squeeze().numpy()
print(speech.shape)
speech_sample = speech[5 * 16000: 20*16000]
# test_dataset =
inputs = processor(speech_sample, sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
print(logits.shape)
print(logits)
predicted_ids = torch.argmax(logits, dim=-1)
print(predicted_ids.shape)
print(predicted_ids)
print("Prediction:", processor.batch_decode(predicted_ids))
# print("Reference:", test_dataset[:2]["sentence"])