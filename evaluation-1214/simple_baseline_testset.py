import speech_recognition as sr
from googletrans import Translator
from pydub import AudioSegment
import torchaudio
import os, glob, json, codecs, math

r = sr.Recognizer()

for name in glob.glob('./MT_data/test/audio/*.wav'):
    json_name = './MT_data/test/' + name.split('\\')[1].split('.')[0] + '.json'
    print(json_name)
    data = []
    with codecs.open(json_name, 'r', "utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    # print(data)


    transcriptions = [rec["transcript"] for rec in data]
    offsets = [rec["offset"] for rec in data]
    durations = [rec["duration"] for rec in data]

    speech_array, sampling_rate = torchaudio.load(name)
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    speech = resampler(speech_array)
    print(speech)

    for i in [1]:
        index_start = math.floor(float(offsets[i]) * 16000)
        index_end = math.ceil((float(durations[i]) + float(offsets[i])) * 16000)
        trimmed_speech = speech[index_start:index_end]

        # print('./MT_data/test/' + name.split('\\')[1].split('.')[0] + '_' + str(i) + '.wav')
        saved_name = './MT_data/test/trimmed/' + name.split('\\')[1].split('.')[0] + '_' + str(i) + '.wav'
        torchaudio.save('./MT_data/test/trimmed/4.wav', trimmed_speech, sampling_rate)

        # with sr.AudioFile(name) as source:
        #     audio = r.record(source)
        #     print(audio)
        # try:
        #     s = r.recognize_google(trimmed_speech, language = 'zh-CN')
        #     translator = Translator()
        #     translation =  translator.translate(s, src='zh-cn', dest='en').text
        #     print("Text: "+ s)
        #     print("Translation:" + translation)
        # except Exception as e:
        #     print("Exception: "+str(e))

# def preprocess_function(examples):
#     inputs = [ex for ex in examples["transcript"]]
#     targets = [ex for ex in examples["translation"]]

#     model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
#     # model_inputs = {}
#     # Set up the tokenizer for targets
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(targets, max_length=max_target_length, truncation=True)

#     model_inputs["labels"] = labels["input_ids"]
#     wav_files = [ex for ex in examples["wav_id"]]
#     wav_offset = [ex for ex in examples["offset"]]
#     wav_duration = [ex if float(ex) > 0.400 else "0.400" for ex in examples["duration"]]
    # wav_files_unique = list(set(wav_files))
    # wav_dic = {}
    # for i in range(len(wav_files_unique)):
    #     speech_array, sampling_rate = torchaudio.load('./data_source/audio_train/' + str(wav_files_unique[i]) + '.wav')
    #     resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    #     speech = resampler(speech_array.squeeze(0)).numpy()
    #     print(speech.shape)
    #     wav_dic[str(wav_files_unique[i])] = speech
#     batch_speech_sample = []
#     for i in range(len(wav_files)):
#         whole_audio = wav_dic[str(wav_files[i])]
#         index_start = math.floor(float(wav_offset[i]) * 16000)
#         index_end = math.ceil((float(wav_duration[i]) + float(wav_offset[i])) * 16000)
#         batch_speech_sample.append(whole_audio[index_start:index_end])
#     audio_batch_inputs = processor(batch_speech_sample, sampling_rate=16_000, return_tensors="pt", padding=True)
#     model_inputs["audio_inputs"] = audio_batch_inputs.input_values.tolist()
#     model_inputs["audio_inputs_mask"] = audio_batch_inputs.attention_mask.tolist()
#     # del model_inputs['input_ids']
#     # del model_inputs['attention_mask']
#     return model_inputs