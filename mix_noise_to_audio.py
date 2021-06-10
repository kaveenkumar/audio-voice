"""
module to mix noise to audio / speech data

two methods of mixing noise
- calculate Additive White Gaussian Noise (AWGN) from given clean audio and then mix it back to original audio
- extract data from a noisy wav file and add it to the original audio
"""

import argparse
import logging
import math
import os
from pathlib import Path
import sys

import librosa
import numpy as np
from scipy.io.wavfile import write

def mix_white_noise(signal, snr_value):
    """
    method1: add additive white gaussian noise
    given a signal file and desired SNR value, this returns the output wav with required AWGN added to the signal
    SNR in dB
    """

    # normalize the signal values
    # find interpolate values between signal.min(), signal.max() as x axis and (-1, 1) as y axis
    signal = np.interp(signal, (signal.min(), signal.max()), (-1, 1))

    # RMS value of signal
    rms_s = math.sqrt(np.mean(signal ** 2))
    # RMS value of noise that depends on SNR value passed
    rms_n = math.sqrt(rms_s ** 2 / (pow(10, int(snr_value) / 10)))
    # because mean=0 for a normal Gaussian distribution, std = rms
    std_n = rms_n
    noise = np.random.normal(0, std_n, signal.shape[0])

    signal_noise = signal + noise

    return signal_noise

  def mix_audio_noise(signal, noise, snr_value):
    """
    method2: add real world noise from a wav file
    given a signal file, noise (audio) and desired SNR value, this gives the noise (scaled version of noise input) that gives the desired SNR
    """

    signal = np.interp(signal, (signal.min(), signal.max()), (-1, 1))
    noise = np.interp(noise, (noise.min(), noise.max()), (-1, 1))

    rms_s = math.sqrt(np.mean(signal ** 2))
    # required RMS of noise that depends on SNR value passed
    rms_n = math.sqrt(rms_s ** 2 / (pow(10, int(snr_value) / 10)))

    # current RMS of noise
    rms_n_current = math.sqrt(np.mean(noise ** 2))
    noise = noise * (rms_n / rms_n_current)

    # helper script to crop the noise file to match input audio file duration
    if len(noise) < len(signal):
        while len(noise) < len(signal):
            noise = np.append(noise, noise)
            noise = noise[0:len(signal)]
    elif len(noise) > len(signal):
        noise = noise[0:len(signal)]

    assert len(noise) == len(signal)

    signal_noise = signal + noise
    # print("SNR = " + str(20 * np.log10(math.sqrt(np.mean(signal ** 2)) / math.sqrt(np.mean(noise ** 2)))))

    return signal_noise

# pylint: disable=too-many-locals
def mix_noise(audio_clean_dir, device_noise_wav, external_noise_dir, snr_value):
    """main function that mixes noise and writes out the augmented audio"""

    for subdir_noise, _, files_noise in os.walk(external_noise_dir):
        for filename_noise in files_noise:
            processed_noise_count = 1
            files_noise_count = len(files_noise)
            filepath_noise = subdir_noise + os.sep + filename_noise
            logger.info(f"NOISE FILE {processed_noise_count} / {files_noise_count} {filename_noise.split('.')[0]}")
            noise, _ = librosa.load(filepath_noise)

            for subdir_clean, _, files_clean in os.walk(audio_clean_dir):
                processed_file_count = 1
                files_clean_count = len(files_clean)
                for filename_clean in files_clean:

                    filepath_clean = subdir_clean + os.sep + filename_clean

                    if filepath_clean.endswith(".wav"):
                        logger.info(f"{processed_file_count} / {files_clean_count} MIXING {filename_noise} FOR AUDIO {filename_clean}")

                        signal, _ = librosa.load(filepath_clean)

                        if device_noise_wav:
                            device_noise = mix_audio_noise(signal, noise, snr_value)
                        else:
                            white_noise = mix_white_noise(signal, snr_value=50)
                            device_noise = white_noise

                        external_noise = mix_audio_noise(signal, noise, snr_value)

                        signal_noise = device_noise + external_noise

                        _, sr = librosa.load(filepath_clean)
                        output_filename = filename_clean.split('.')[0] + "_" + filename_noise.split('.')[0] + "_snr" + str(snr_value) + ".wav"
                        write(Path(audio_clean_dir, output_filename), sr, signal_noise)
                    processed_file_count += 1

                    
if __name__ == '__main__':
    # args = get_args()
    # mix_noise(args)
