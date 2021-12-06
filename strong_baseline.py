# Needed:
# !pip install transformers
# !pip install sentencepiece

from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import pipeline

model = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", return_tensors="pt")
translation = pipeline("translation_chinese_to_english", model=model, tokenizer=tokenizer)

import os, glob
import codecs, json

path = './'

for input_filename in glob.glob(os.path.join(path, '*.json')):
  with codecs.open(input_filename, "r", "utf-8") as input:
    output_filename = input_filename[:-5] + "_result.json"
    print(output_filename)
    with codecs.open(output_filename, 'w', "utf-8") as output:
      i = 0
      for row in input:
        i += 1
        print(i)
        data = json.loads(row)
        translated_text = translation(data['transcript'], max_length=70)[0]['translation_text']
        lineInfo = {"transcript": data['transcript'], "translation": translated_text, "reference": data['translation']}
        output.write(json.dumps(lineInfo) + "\n")

       