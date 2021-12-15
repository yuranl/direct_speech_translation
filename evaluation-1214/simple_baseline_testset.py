import speech_recognition as sr
from googletrans import Translator
from pydub import AudioSegment
import torchaudio
import glob, json, codecs, os
from google.cloud import translate_v2 as translate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./stone-airfoil-335221-d8728d7be624.json"

translate_client = translate.Client()
r = sr.Recognizer()

for name in glob.glob('./evaluation-1214/audio_train/*.wav'):
    json_name = './evaluation-1214/' + name.split('\\')[1].split('.')[0] + '.json'
    # print(json_name)
    data = []
    with codecs.open(json_name, 'r', "utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    # print(data)

    transcriptions = [rec["transcript"] for rec in data]
    offsets = [rec["offset"] for rec in data]
    durations = [rec["duration"] for rec in data]

    speech = AudioSegment.from_wav(name)
    json_file_path = './evaluation-1214/result_jsons/' + name.split('\\')[1].split('.')[0] + '_result.json'
    print(json_file_path)
    translator = Translator()

    for i in range(len(offsets)):
        if i % 50 == 0:
            translator = Translator()
        t1 = float(offsets[i]) * 1000
        t2 = (float(durations[i]) + float(offsets[i])) * 1000
        trimmed = speech[t1:t2]

        saved_name = './evaluation-1214/audio_train/trimmed/' + name.split('\\')[1].split('.')[0] + '_' + str(i) + '.wav'
        trimmed.export(saved_name, format="wav")
    
        f = codecs.open(json_file_path, 'a', "utf-8")
        with sr.AudioFile(saved_name) as source:
            audio = r.record(source)
            try:
                s = r.recognize_google(audio, language = 'zh-CN')
            except Exception as e:
                s = ''
            print(s)
            try:
                translation = translate_client.translate(s, target_language='en')['translatedText']
            except Exception as e:
                translation = ''
            print(translation)

        json_line = {}
        json_line["translation"] = translation
        json_line = json.dumps(json_line)
        f.write(json_line + '\n')
        f.close()
        # print('./MT_data/test/' + name.split('\\')[1].split('.')[0] + '_' + str(i) + '.wav')


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