#!/usr/bin/env python3.9

# Standard Python.
import os
import pickle

from glob import glob
from optparse import OptionParser

# Numpy.
import numpy as np

# Machine learning toolkit.
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import *

# Sound recording.
import sounddevice as sd

# Fun big fonts.
from pyfiglet import Figlet

# Local
from sound_sample import SoundSample

import scipy

def arg_parse(parser):
    parser.add_option('-m', '--model', dest = 'model_file',  help = 'Model to read from.');

    return parser.parse_args();

def main():
    parser = OptionParser();
    (options, args) = arg_parse(parser);

    if (not options.model_file):
        parser.error("Model file is required: see --help for details.");

    print("Loading model from:", options.model_file);

    model_in = open(options.model_file, "rb");
    dt = pickle.load(model_in);
    model_in.close();

    f = Figlet(font = 'standard');

    # SoundDevice constants.
    SD_FS = 44100;
    SD_DURATION = 0.5;
    SD_BUFFER_DUR = 1;
    SD_CHANNELS = 1;
    SD_SAMPLES_PER_ITER = int(SD_DURATION * SD_FS);

    print("Filling audio buffer...");
    pcm = sd.rec(int(SD_BUFFER_DUR * SD_FS), samplerate = SD_FS, channels = SD_CHANNELS, dtype = 'float32');
    sd.wait();

    print(len(pcm), "floats in pcm buffer.");

    print("Now classifying...");

    while True:
        buf = sd.rec(SD_SAMPLES_PER_ITER, samplerate = SD_FS, channels = SD_CHANNELS, dtype = 'float32');
        sd.wait();

        # Add new samples and chuck old ones.
        pcm = np.concatenate((pcm, buf));
        pcm = pcm[SD_SAMPLES_PER_ITER:];

        # No idea why we have to do this, but FFT is just completely wrong if we don't read from a file.
        scipy.io.wavfile.write("/tmp/tmp.wav", SD_FS, pcm);
        sample = SoundSample("/tmp/tmp.wav", None);
        sample.preprocess();

        #sample.plots();
        #break;

        # Use the model.
        fv = [];
        fv.append(sample.features(method = "domain", method_args = (10,), windowing = False));
        scaler = Normalizer();
        fv = scaler.fit_transform(fv);

        print(fv);

        prediction = dt.predict(fv);

        print(f.renderText(prediction[0]));

if __name__ == "__main__":
    main();
