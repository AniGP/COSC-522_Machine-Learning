#!/usr/bin/env python3.9

# Standard Python.
import os
import pickle

from glob import glob
from optparse import OptionParser

# Numpy
import numpy as np

# Machine learning toolkit.
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

# Local
from sound_sample import SoundSample

def arg_parse(parser):
    parser.add_option('-o', '--model-output',     dest = 'model_file',  help = 'Filename to write generated model to.');
    parser.add_option('-s', '--training-samples', dest = 'samples_dir', help = 'Directory to load training samples from. Classes must be in separte folders.');

    return parser.parse_args();

def load_samples(sample_directory):
    class_dirs = glob(sample_directory + "/*")

    samples = [];

    for class_dir in class_dirs:
        label = os.path.basename(class_dir);
        class_dir_glob = glob(class_dir + "/*");

        print("Reading files for label '" + label + "'");

        for sample_file in class_dir_glob:
            sample = SoundSample(sample_file, label);
            sample.preprocess();

            samples.append(sample);

    return samples;

def main():
    parser = OptionParser();
    (options, args) = arg_parse(parser);

    if (not options.model_file):
        parser.error("Output model file is required: see --help for details.");
    if (not options.samples_dir):
        parser.error("Sample directory is required: see --help for details.");

    print("Loading samples from:", options.samples_dir);

    samples = load_samples(options.samples_dir);

    print("Got", len(samples), "samples.");

    # Get data / label vectors.
    fv = [];
    labels = [];

    print("Extracting features...");

    for sample in samples:
        sample_features = sample.features(method = "domain", method_args = (10,), windowing = True);

        #fv.append(sample_features);
        #labels.append(sample.label);

        # We have a bunch of windows now: 3 features per window. Break it up and add it to the fv.
        for i in range(0, len(sample_features), 3):
            fv.append(sample_features[i:i + 3]);
            labels.append(sample.label);

    # Scale.
    scaler = Normalizer();
    fv = scaler.fit_transform(fv);

    print("Training model...");

    # Train.
    # We don't actually split it here, we want to use all data for our model in this case.
    dt = RandomForestClassifier();
    dt.fit(fv, labels);

    # Testing the model.
    cv_scores = cross_val_score(dt, fv, labels, cv = 10)

    print("Domain features WITH windowing.");
    print('Average Cross Validation Score from Training:', cv_scores.mean(), sep = '\n', end = '\n')

    out = open(options.model_file, "wb");
    pickle.dump(dt, out);
    out.close();

    print("Wrote model to '" + options.model_file + "'");

if __name__ == "__main__":
    main();
