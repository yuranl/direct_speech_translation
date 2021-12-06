import os
import glob
import codecs
import json

###############################
####change below parameters####

input_format = ".json"
#mode = "ref" #reference speech
mode = "tst" #output speech to be evaluated
#mode = "src" #source speech
setid = "1" #id used to identify this set of data

#input_path = "data/source/"
input_path = "data/translation/"
#input_path = "data/reference/"



###############################
####start of script        ####

output_filename = mode + setid + ".sgm"
input_filename_list = glob.glob(input_path + "*" + input_format)
print("Files found at " + input_path + "*" + input_format + ":")
print(input_filename_list)

with codecs.open(output_filename, 'w', "utf-8") as output:
    output.write("<" + mode + "set setid=\"" + setid + "\" srclang =\"chs\" trglang = \"eng\">\n")
    output.write("<DOC docid=\"doc\" genre=\"text\" sysid=\"" + mode + setid + "\">\n")
    seg_id = 0
    for input_filename in input_filename_list:
        with codecs.open(input_filename, "r", "utf-8") as input:
            seg_id += 1
            print("Processing file: " + str(seg_id) + "; File name: " + input_filename + "\n")

            text = ""
            for row in input:
                data = json.loads(row)
                if mode == "src":
                    text += data["transcript"]
                else:
                    text += data["translation"]

            #data = json.load(input)
            #text = ""

            #for row in data:
            #    if mode == "src":
            #        text += row["transcript"]
            #    else:
            #        text += row["translation"]

            #print(text)
            output.write("<seg id=" + str(seg_id) + ">" + text + "</seg>\n")

    output.write("</DOC>\n")
    output.write("</" + mode + "set>")


