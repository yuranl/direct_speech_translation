from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os, glob
import numpy as np

path = './'
all_mfcc = []

for filename in glob.glob(os.path.join(path, '*.wav')):
    (rate,sig) = wav.read(filename)
    mfcc_feat = mfcc(sig,rate,winstep=0.015,nfft=2048)
    print(mfcc_feat)
    all_mfcc.append(mfcc_feat)
    print(len(mfcc_feat))