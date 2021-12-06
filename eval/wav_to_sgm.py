import os
import glob
import speech_recognition as sr
import codecs

###############################
####change below parameters####

input_format = ".wav"
#input_format = ".mp3"
mode = "ref" #reference speech
#mode = "tst" #output speech to be evaluated
setid = "1" #id used to identify this set of data

#input_path = ""
input_path = "output/"
#input_path = "reference/"



###############################
####start of script        ####

output_filename = mode + setid + ".sgm"
input_filename_list = glob.glob(input_path + "*" + input_format)
print("Files found at " + input_path + "*" + input_format + ":")
print(input_filename_list)
r = sr.Recognizer()

with codecs.open(output_filename, 'w', "utf-8") as output:
    output.write("<" + mode + "set setid=\"" + setid + "\" srclang =\"chs\" trglang = \"eng\">\n")
    output.write("<DOC docid=\"doc\" genre=\"text\" sysid=\"" + mode + setid + "\">\n")
    seg_id = 0
    for input_filename in input_filename_list:
        with sr.AudioFile(input_filename) as input:
            seg_id += 1
            print("Processing file: " + str(seg_id) + "; File name: " + input_filename + "\n")
            audio = r.record(input)
            text = r.recognize_google(audio, language = 'zh-CN', show_all = True)
            #print(text)
            #print(text["alternative"][0]["transcript"])
            output.write("<seg id=" + str(seg_id) + ">" + text["alternative"][0]["transcript"] + "</seg>\n")
            #try:
            #    text = r.recognize_google(audio, language = 'en-US')
            #    print(text)
            #except Exception as e:
            #    print("Exception: "+str(e))
        #output.write("<seg id=" + str(seg_id) + ">" + text + "</seg>\n")
    output.write("</DOC>\n")
    output.write("</" + mode + "set>")


