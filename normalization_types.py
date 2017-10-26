from __future__ import print_function
import librosa
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import librosa.display as disp
import librosa.feature as feat
import numpy as np
import scipy.signal as sig
import soundfile as sf
import numpy.fft as fft
#pathToFile = "audiofiles/single_cry1.wav"
pathToFile = "audiofiles/cry_clean02_corridor_50db.wav"

frameLength = 1024
hopLength = frameLength/2

def visualize(y,sr,*args):

    support = np.linspace(0, librosa.get_duration(y=y, sr=sr), np.size(y))

    panels = 1+len(args)
    plt.figure()
    plt.subplot(panels, 1, 1)
    plt.plot(support, y, linewidth=0.1)

    counter = 2
    for P in args:
        plt.subplot(panels, 1, counter)
        counter = counter+1
        disp.specshow(librosa.core.power_to_db(P), sr=sr, hop_length=hopLength, x_axis="time", y_axis="linear")
        plt.colorbar(format="%+2.0f db")

    plt.show()

def calPowerSpectrogram(y):
    S = librosa.core.stft(y, n_fft=frameLength, hop_length=hopLength, win_length=frameLength,window=sig.hamming(frameLength), center=False)
    P = np.abs(S)**2
    return P

def inverseAndWrite(y, sr, *args):
    support = np.linspace(0, librosa.get_duration(y=y, sr=sr), np.size(y))
    panels = 1 + len(args)
    plt.figure()
    plt.subplot(panels, 1, 1)
    plt.plot(support, y, linewidth=0.1)

    counter = 2
    for P in args:
        S = np.sqrt(P)
        y = librosa.istft(S, hop_length=hopLength, win_length=frameLength, window=sig.hamming(frameLength), center=False)
        y = np.clip(y, -0.5, 0.5)
        support = np.linspace(0, librosa.get_duration(y=y, sr=sr), np.size(y))
        plt.subplot(panels, 1, counter)
        plt.plot(support, y, linewidth=0.1)
        counter = counter+1
        name = str(counter) + ".wav"
        #sf.write(name, y, sr, subtype='PCM_16')

    plt.show()

def main():
    y, sr = librosa.load(pathToFile)
    y_1 = librosa.util.normalize(y)
    y_2 = y*0.1
    y = np.concatenate((y_1, y_2), axis=0)
    P = calPowerSpectrogram(y)

    P_framelength = P/frameLength
    P_pdf = P / P.sum(axis=0)
    P_frameNormed = librosa.util.normalize(P, axis=0)

    visualize(y, sr, P, P_framelength, P_pdf, P_frameNormed)

    #inverseAndWrite(y, sr, P, P_framelength*frameLength,P_pdf*frameLength, P_frameNormed*frameLength)

main()






