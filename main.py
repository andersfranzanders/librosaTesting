from __future__ import print_function
import librosa
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import librosa.display as disp
import librosa.feature as feat
import numpy as np
import scipy.signal as sig
import numpy.fft as fft
#pathToFile = "audiofiles/single_cry1.wav"
pathToFile = "audiofiles/cry_clean01_barcelona_-5db.wav"

frameLength = 1024
hopLength = frameLength/2

def main():


    y, sr = librosa.load(pathToFile)
    y = librosa.util.normalize(y)

    y_framed = librosa.util.frame(y,frame_length=frameLength,hop_length=hopLength)
    y_1 = y_framed[:,1]

    y_rmses = librosa.feature.rmse(y, frame_length=frameLength, hop_length=hopLength)
    zcr = librosa.feature.zero_crossing_rate(y,frame_length=frameLength,hop_length=hopLength)

    S = librosa.core.stft(y, n_fft=frameLength, hop_length=hopLength, win_length=frameLength, center=True)
    P = np.abs(S)**2
    P = P/P.sum(axis=0)

    #P = librosa.util.normalize(P_old, axis=1)
    M = feat.melspectrogram(S=P, sr=sr, n_mels=40)
    MFCCs = feat.mfcc(S=librosa.power_to_db(M), n_mfcc=12)
    #MFCCs2 = feat.mfcc(y=y, sr=sr, n_mfcc=12, n_mels=60, n_fft=frameLength, hop_length=hopLength)

    print(MFCCs)
    #print(MFCCs2)

    frame_support = np.linspace(0, librosa.get_duration(y=y, sr=sr), np.size(y_rmses))
    support = np.linspace(0,librosa.get_duration(y=y, sr=sr),np.size(y))
    #print(support)


    plt.figure()
    plt.subplot(4,1,1)
    #disp.waveplot(y, sr=sr)
    plt.plot(support, y, linewidth=0.1)
    plt.plot(frame_support, y_rmses[0,:],linewidth=1)
    plt.subplot(4,1,2)
    disp.specshow(librosa.core.power_to_db(P),sr=sr, hop_length=hopLength,x_axis="time",y_axis="linear")
    plt.colorbar(format="%+2.0f db")
    plt.subplot(4, 1, 3)
    disp.specshow(librosa.core.power_to_db(M, ref=frameLength), sr=sr, hop_length=hopLength, y_axis='mel', x_axis="time")
    #plt.colorbar(format="%+2.0f db")
    plt.subplot(4, 1, 4)
    librosa.display.specshow(MFCCs, x_axis='time')
    #plt.colorbar()



    plt.show()

print("hello!")
main()
