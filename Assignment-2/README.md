### Assignment 2: Using Sound to Detect Activities and Events

- Develop a machine learning pipeline to detect activities and events using sound. The assignment will involve data collection, data pre-processing/signal conditioning, feature extraction, using an existing ML implementation, and analysis of results.

**Data Collection (5% grade):**

Collect 20 samples each for 5 classes:

- Microwave (run for 30 seconds, and I would suggest to include door opening, closing, and beeps as part of each recording)
- Blender (run for 30 seconds)
- Fire alarm or any other kind of siren
- Vacuum Cleaner (run for approx. 30 seconds and perhaps move the vacuum cleaner around as it will change the sound profile a bit)
- Music (approx 30 seconds for each sample, the music of your choice). Try varying the song (e.g., 5 songs with 4 samples each)

- For any device that you might not have (e.g., please don’t trigger an actual fire alarm), find a recording on the Internet (maybe on YouTube) and record its sound on your phone. Make sure not to use the audio file directly off the Internet. Make your own recording of the audio file because you want the general variability between recordings for your 20 samples. You do not need to choose 20 different examples of a sound. For example, if you don’t have access to a blender, don’t search for 20 blender sounds on the Internet. Find one sound and record it 20 times. Identifying 20 different blenders as “blender” is a much harder problem for a course homework.

- For recording the sounds, you can use these apps (feel free to try out others but make sure that you are recording uncompressed WAV files):

- iOS: Voice Record, Android: Wav Recorder

- I used Audio Recorder - WAV, M4A on my iOS device to capture the data.

- In a real-world scenario, where a system like this runs the whole time to detect different events/sounds, the system would need to filter out silent periods. Thus, in addition to the 5 event classes, also record 20 samples of silence (approx 30 seconds). These recordings will be used to develop a logic that can be used in the future by someone to filter out silent periods. Now, it is entirely up to you whether you want to treat these silent files as a separate sixth class in your ML pipeline or if you want to filter these out in the data pre-processing.

**Pre-processing (5% grade):**

- You will need to process your collected data in some way before extracting features from it. For example, calculating FFT to convert the raw time-domain signal into frequency-domain, or removing some frequency bands that you think might not contain anything useful.

**Feature Engineering/Extraction and ML algorithm (60% total):**

- You are free to use any ML algorithm with any parameter and configs you like. I would suggest to try different algorithms and see what works well.

For features, you will try two approaches:

- Binning the spectrogram data from the recordings and using each bin as a feature. E.g., if you have a 1024-point FFT of a recording, then your FFT output will be a [1024 x num_of_windows] samples. You want to convert this 2D array of samples into a smaller array. You can use the sample code I provided in the shared Dropbox folder to bin the values. Feel free to experiment with different sizes of bins. Read some of the papers we discussed in the class for inspiration.
Extracting domain-specific features. Find specific phenomena for each class that you want to capture. These features can be in the time or frequency domain.

For calculating your features (binned or domain-specific), you need to choose a window of data. You will try two approaches here, as well:

- Treating the whole approx. 30 seconds recording as a single “window.”
- Dividing each recording into multiple windows. Feel free to experiment with different window sizes and overlaps. As a convention, though, I would suggest using a 50% overlap between windows.

**Analysis: Analyze the pipeline’s performance using 10-fold cross-validation.**

- Performance: Aim for above 80% performance in at least 3 cases, and above 90% in at least 1. i.e., it is okay if the classification accuracy is below 80% for one of the cases. However, these performance thresholds are not rigid. Each of you is collecting your own dataset, and I understand that the performance might vary. The performance targets here are merely guidelines. You will be judged more on the soundness and rationale of your approach.

**Write-up (30% grade):**

You will submit a write-up (approx 2 pages) explaining:

- Data collection process
- Rationale for features (no need to explain the bin sizes, explain the domain-specific features or if you end up doing any other feature engineering or anything else that you feel like sharing)
- Graph and describe results for different conditions.
- Refer to some of the shared papers as templates for the write-up. This write-up does not need to be too long.