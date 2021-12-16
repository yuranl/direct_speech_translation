import speech_recognition as sr
from googletrans import Translator
from pydub import AudioSegment
import torchaudio
import glob, json, codecs, os
from google.cloud import translate_v2 as translate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./stone-airfoil-335221-d8728d7be624.json"

translate_client = translate.Client()
r = sr.Recognizer()

for name in glob.glob('./MT_data/test/audio/*.wav'):
    json_name = './MT_data/test/' + name.split('\\')[1].split('.')[0] + '.json'
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
    json_file_path = './evaluation-1214/baseline_model_output/' + name.split('\\')[1].split('.')[0] + '_result.json'
    print(json_file_path)
    translator = Translator()

    for i in range(len(offsets)):
        if i % 50 == 0:
            translator = Translator()
        t1 = float(offsets[i]) * 1000
        t2 = (float(durations[i]) + float(offsets[i])) * 1000
        trimmed = speech[t1:t2]

        saved_name = './evaluation-1214/trimmed/' + name.split('\\')[1].split('.')[0] + '_' + str(i) + '.wav'
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
