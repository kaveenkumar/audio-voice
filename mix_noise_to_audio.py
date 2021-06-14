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

logger = logging.getLogger('noise_mixing')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s; %(pathname)s; %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting to mix noise to the train dataset')


def get_args():
    """ get args from stdin"""

    parser = argparse.ArgumentParser(
        description="""Mix selected noise sources to the input clean audio files.
        The amount of noise mixed to the clean audio is decided by the SNR value""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    parser.add_argument("--input_directory", type=str, default=None,
                        help="Choose the input directory that contains clean audio files")

    parser.add_argument("--device_noise", type=str, default=None,
                        help="Select an input wav file that contains the device internal recording")

    parser.add_argument("--noise_directory", type=str, default=None,
                        help="Choose the noise directory that contains different noise sources")

    parser.add_argument("--snr", type=int, dest='snr', default=20,
                        help="Enter an SNR value for the noise levels to be mixed")

    print(' '.join(sys.argv))
    print(sys.argv)

    args = parser.parse_args()
    return args


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

    # for each noise file in the noise directory, load them over iterations
    for subdir_noise, _, files_noise in os.walk(external_noise_dir):
        processed_noise_count = 1
        for filename_noise in files_noise:
            files_noise_count = len(files_noise)
            filepath_noise = subdir_noise + os.sep + filename_noise
            logger.info(f"NOISE FILE {processed_noise_count} / {files_noise_count} {filename_noise.split('.')[0]}")
            noise, _ = librosa.load(filepath_noise)

            # for each clean audio in input directory, load them over iterations
            for subdir_clean, _, files_clean in os.walk(audio_clean_dir):
                processed_file_count = 1
                files_clean_count = len([file for file in files_clean if 'snr' not in file])
                for filename_clean in files_clean:

                    filepath_clean = subdir_clean + os.sep + filename_clean

                    if (filepath_clean.endswith(".wav")) and ('snr' not in filepath_clean):
                        logger.info(f"{processed_file_count} / {files_clean_count} MIXING {filename_noise} FOR AUDIO {filename_clean}")

                        signal, _ = librosa.load(filepath_clean)

                        # add AWGN - might be useful for devices with high internal noise
                        if device_noise_wav:
                            # mix device noise if provided
                            device_noise = mix_audio_noise(signal, noise, snr_value)
                        else:
                            # mix white noise if device noise not provided
                            white_noise = mix_white_noise(signal, snr_value=50)
                            device_noise = white_noise

                        # mix real world noise
                        external_noise = mix_audio_noise(signal, noise, snr_value)

                        # add device noise + real world noise
                        signal_noise = device_noise + external_noise

                        # export the augmented audio
                        _, sr = librosa.load(filepath_clean)
                        output_filename = filename_clean.split('.')[0] + "_" + filename_noise.split('.')[0] + "_snr" + str(snr_value) + ".wav"
                        write(Path(audio_clean_dir, output_filename), sr, signal_noise)
                        processed_file_count += 1
            processed_noise_count += 1


if __name__ == '__main__':
    audio_clean_directory = sys.argv[1]
    device_noise_file = sys.argv[2]
    external_noise_directory = sys.argv[3]
    snr = sys.argv[4]
    mix_noise(audio_clean_directory, device_noise_file, external_noise_directory, snr)

    # args = get_args()
    # mix_noise(args)
