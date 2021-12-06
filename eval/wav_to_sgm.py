import os
import glob
import speech_recognition as sr

###############################
####change below parameters####

input_format = ".wav"
#input_format = ".mp3"
mode = "ref" #reference speech
#mode = "tst" #output speech to be evaluated
setid = "1" #id used to identify this set of data

input_path = "/output/"
#input_path = "/reference/"



###############################
####start of script        ####

output_filename = mode + setid + ".sgm"
input_filename_list = glob.glob(input_path + "*" + input_format)
r = sr.Recognizer()

with open(output_filename, 'w') as output:
    output.write("<" + mode + "set setid=\"" + setid + "\" srclang =\"chs\" trglang = \"eng\">\n")
    output.write("<DOC docid=\"doc\" genre=\"text\" sysid=\"" + mode + setid + "\">\n")
    seg_id = 0
    for input_filename in input_filename_list:
        with open(input_filename) as input:
            seg_id += 1
            print("Processing file: " + seg_id + "; File name: " + input_filename + "\n")
            audio = r.record(input)
            try:
                text = r.recognize_google(audio, language = 'en-US')
                output.write("<seg id=" + seg_id + ">" + text + "</seg>\n")
            except Exception as e:
                print("Exception: "+str(e))
    output.write("</DOC>\n")
    output.write("</" + mode + "set>")


