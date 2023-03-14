#!/usr/bin/env python3.9

# Numpy, always.
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Need plots.
import matplotlib.pyplot as plt

# Scipy for fft's and the like.
import scipy as sc
import scipy.io.wavfile as wavfile
from scipy import signal
from scipy.fftpack import fft, fftfreq
from scipy import stats

# LibROSA, loads .wav files / some signal processing.
import librosa
import librosa.display

# CV2, used for binning.
import cv2

# Ipython for basic visual output types.
import IPython

# Standard Python libs.
import os
import glob

# Class definitions
class SoundSample:
    def __init__(self, file = None, label = None):
        self.file  = file;
        self.label = label;

        self.fft_data     = None;
        self.spectro_data = None;

        if (self.file == None):
            return;

        x, fs = librosa.load(self.file, sr = None, mono = True, offset = 0.0, duration = None);

        self.x     = x;
        self.fs    = fs;

    def from_pcm(self, x, fs, label = None):
        self.file  = None;
        self.label = label;

        self.x     = x;
        self.fs    = fs;

        self.fft_data     = None;
        self.spectro_data = None;

    def __domain_features(self, window_count, windowing):
        fv = [];
        n_point = 1024;

        if (windowing):
            window_size = int(self.x.size / ((window_count + 1) / 2));
            overlap = int(window_size / 2);

            # Creates window_count views with half overlap.
            views = sliding_window_view(self.x, window_size)[::overlap];

            for view in views:
                # Same features as no-windowing.
                snr = view.mean() / view.std();
                fv.append(snr);

                mfccs = librosa.feature.mfcc(y = view, sr = self.fs, n_mfcc=40);
                fv.append(mfccs.T.mean());

                sbw = librosa.feature.spectral_bandwidth(y = view, sr = self.fs)[0].mean();
                fv.append(sbw);

        else:
            # Signal-to-Noise ratio feature.
            snr = self.x.mean() / self.x.std();
            fv.append(snr);

            # Mel-Freq Cepstral Coeff. (see: https://librosa.org/doc/main/generated/librosa.feature.mfcc.html)
            mfccs = librosa.feature.mfcc(y = self.x, sr = self.fs, n_mfcc=40);
            fv.append(mfccs.T.mean());

            # Spectral Bandwidth (see: https://librosa.org/doc/main/generated/librosa.feature.spectral_bandwidth.html)
            sbw = librosa.feature.spectral_bandwidth(y = self.x, sr = self.fs)[0].mean();
            fv.append(sbw);

        return fv;

    def __spectro_features(self, freq_bins, time_bins, windowing):
        fv = [];

        if (windowing):
            # Spectrogram basically handles the windowing for us.
            # FFT over n_point size windows.
            f, t, pxx, n_point = self.spectrogram();

            binned_pxx = cv2.resize(np.log10(pxx), (time_bins, freq_bins));

            #plt.pcolormesh(np.log10(binned_pxx));

            for x_bin in binned_pxx:
                for y_bin in x_bin:
                    fv.append(y_bin);
        else:
            # We get a fft of the whole signal - i.e. one window.
            freq, mag, n_point = self.fft();

            fbins = np.array_split(mag, freq_bins);
            for fbin in fbins:
                fv.append(fbin.mean());

        return fv;

    def features(self, method, method_args, windowing):
        print("Extract features from:", self.file if (self.file) else "PCM");

        fv = [];

        if (method == "domain"):
            fv = self.__domain_features(method_args[0], windowing);
        elif (method == "spectro"):
            fv = self.__spectro_features(method_args[0], method_args[1], windowing);
        else:
            print("Unknown method '" + method + "'! Use one of {domain, spectro}.");

        return fv;

    def preprocess(self):
        self.apply_lpf(15000);

    def plots(self):
        lab = self.label if (self.label) else "PCM";

        # Raw wav data.
        plt.figure(figsize = (12, 8));
        plt.plot(np.linspace(0, len(self.x) / self.fs, num = len(self.x)), self.x);
        plt.title("Signal - Unfiltered (label: " + lab + ")");
        plt.xlabel("t");
        plt.ylabel("x(t)");
        plt.show();

        # FFT
        freq, mag, n_point = self.fft();
        plt.figure(figsize = (12, 8));
        plt.grid();
        plt.plot(freq, mag);
        plt.fill_between(freq, mag);
        plt.title("Signal Spectrum (1024-point) - Unfiltered Magnitude (label: " + lab + ")")
        plt.xlabel("f (Hz)");
        plt.ylabel("X(jw)");
        plt.show();

        # Spectrogram
        freq, time, spectro, n_point = self.spectrogram();
        plt.figure(figsize = (12, 8));
        plt.pcolormesh(time, freq, np.log10(spectro));
        plt.title("Signal Spectrogram (1024-point) - Unfiltered (label: " + lab + ")")
        plt.xlabel("t");
        plt.ylabel("f (Hz)");
        plt.show();

    def play(self):
        print("Audio sample: " + self.file);

        display(IPython.display.Audio(data = self.x, rate = self.fs));

    def spectrogram(self, n_point = 1024):
        if (self.spectro_data == None or n_point != self.spectro_data[3]):
            freq, time, sxx = signal.spectrogram(self.x, fs = self.fs, nperseg = n_point, noverlap = n_point / 2);
            self.spectro_data = (freq, time, sxx, n_point);

        return self.spectro_data;

    def fft(self, n_point = 1024):
        if (self.fft_data == None or n_point != self.fft_data[2]):
            freq = fftfreq(n = n_point, d = (1 / self.fs))[:int(n_point / 2)];
            fft_y = fft(self.x, n_point)[:int(n_point / 2)];
            mag = np.abs(fft_y);

            self.fft_data = (freq, mag, n_point);

        return self.fft_data;

    def apply_lpf(self, cutoff_freq, lpf_order = 4):
        num, denom = signal.butter(lpf_order, cutoff_freq, fs = self.fs, btype = 'lowpass', analog = False);
        self.x = signal.lfilter(num, denom, self.x);

        # Reset cached data.
        self.fft_data     = None;
        self.spectro_data = None;

        return self.x;
