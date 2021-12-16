import os, glob

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
        self.conv1 = nn.Conv1d(21128, 512, 5, 5,bias=True)
        self.relu = nn.ReLU()
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained("./finetuned_model/1213 wd0002/3")

    def forward(self, batch):
        del batch['input_ids']
        del batch['attention_mask']
        logits = self.asr_model(batch['audio_inputs'], attention_mask=batch['audio_inputs_mask']).logits
        # translation_input = self.conv2(self.relu(self.conv1(logits.transpose(1, 2))))
        translation_input = self.conv1(logits.transpose(1, 2))
        translation_input = self.relu(translation_input)
        length = translation_input.shape[2]
        batch_size = translation_input.shape[0]
        new_attention_mask = torch.ones(batch_size,length, device=translation_input.device)
        outputs = self.translation_model(inputs_embeds=translation_input.transpose(1,2), attention_mask = new_attention_mask, labels=batch.labels)
        return outputs

    def generate(self, batch, max_length=512):
        del batch['input_ids']
        del batch['attention_mask']
        logits = self.asr_model(batch['audio_inputs'], attention_mask=batch['audio_inputs_mask']).logits
        translation_input = self.conv1(logits.transpose(1, 2))
        translation_input = self.relu(translation_input)
        length = translation_input.shape[2]
        batch_size = translation_input.shape[0]
        new_attention_mask = torch.ones(batch_size,length, device=translation_input.device)
        encoder_output = self.translation_model.get_encoder()(inputs_embeds=translation_input.transpose(1,2))
        generated_tokens = self.translation_model.generate(encoder_outputs = encoder_output, attention_mask = new_attention_mask, max_length = max_length)
        return generated_tokens

split_datasets  = load_dataset('json', data_files={'train': './transcription_translation/6.json', 'validation': './translation_validation/102371.json'})
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")


for name in glob.glob('./MT_data/test/audio/*.wav'):
    

# with sr.AudioFile('chunk2.wav') as source:
#     audio = r.record(source)
# try:
#     s = r.recognize_google(audio, language = 'zh-CN')
#     translator = Translator()
#     translation =  translator.translate(s, src='zh-cn', dest='en').text
#     print("Text: "+ s)
#     print("Translation:" + translation)
#     tts = gTTS(translation)
#     tts.save('translated.mp3')
#     # sound = AudioSegment.from_mp3(os.getcwd()+ "//translated.mp3")
#     # sound.export("translated.wav", format="wav")
# except Exception as e:
#     print("Exception: "+str(e))

# import speech_recognition as sr
# import os
# from pydub import AudioSegment
# from pydub.silence import split_on_silence
#
# # create a speech recognition object
# r = sr.Recognizer()
#
# # a function that splits the audio file into chunks
# # and applies speech recognition
# def get_large_audio_transcription(path):
#     """
#     Splitting the large audio file into chunks
#     and apply speech recognition on each of these chunks
#     """
#     # open the audio file using pydub
#     sound = AudioSegment.from_wav(path)
#     # split audio sound where silence is 700 miliseconds or more and get chunks
#     chunks = split_on_silence(sound,
#         # experiment with this value for your target audio file
#         min_silence_len = 500,
#         # adjust this per requirement
#         silence_thresh = sound.dBFS-14,
#         # keep the silence for 1 second, adjustable as well
#         keep_silence=500,
#     )
#     folder_name = "audio-chunks"
#     # create a directory to store the audio chunks
#     if not os.path.isdir(folder_name):
#         os.mkdir(folder_name)
#     whole_text = ""
#     # process each chunk
#     for i, audio_chunk in enumerate(chunks, start=1):
#         # export audio chunk and save it in
#         # the `folder_name` directory.
#         chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
#         audio_chunk.export(chunk_filename, format="wav")
#         # recognize the chunk
#         with sr.AudioFile(chunk_filename) as source:
#             audio_listened = r.record(source)
#             # try converting it to text
#             try:
#                 text = r.recognize_google(audio_listened, language = 'zh-CN')
#             except sr.UnknownValueError as e:
#                 print("Error:", str(e))
#             else:
#                 text = f"{text.capitalize()}. "
#                 print(chunk_filename, ":", text)
#                 whole_text += text
#     # return the text for all chunks detected
#     return whole_text
#
# get_large_audio_transcription('4.wav')