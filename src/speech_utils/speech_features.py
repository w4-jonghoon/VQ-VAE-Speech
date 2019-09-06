 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-Speech.                                               #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

import librosa
import numpy as np
from python_speech_features.base import mfcc, logfbank
from python_speech_features import delta


class SpeechFeatures(object):

    default_rate = 16000
    default_filters_number = 13
    default_augmented = True

    @staticmethod
    def mfcc(signal, rate=default_rate, filters_number=default_filters_number, augmented=default_augmented):
        mfcc_features = mfcc(signal, rate, numcep=filters_number)
        if not augmented:
            return mfcc_features
        d_mfcc_features = delta(mfcc_features, 2)
        a_mfcc_features = delta(d_mfcc_features, 2)
        concatenated_features = np.concatenate((
                mfcc_features,
                d_mfcc_features,
                a_mfcc_features
            ),
            axis=1
        )
        return concatenated_features

    @staticmethod
    def logfbank(signal, rate=default_rate, filters_number=default_filters_number, augmented=default_augmented):
        logfbank_features = logfbank(signal, rate, nfilt=filters_number)
        if not augmented:
            return logfbank_features
        d_logfbank_features = delta(logfbank_features, 2)
        a_logfbank_features = delta(d_logfbank_features, 2)
        concatenated_features = np.concatenate((
                logfbank_features,
                d_logfbank_features,
                a_logfbank_features
            ),
            axis=1
        )
        return concatenated_features

    @staticmethod
    def spectrogram(signal, rate=default_rate, filters_number=default_filters_number, augmented=default_augmented):
        signal = signal.numpy()
        length = signal.shape[2]
        signal = signal.reshape((length,))
        AUDIO_PARAMS = {
            'audio_sample_rate': rate,
            'audio_preemphasis': 0.97,
            'audio_frame_length': 50.0,
            'audio_frame_step': 12.5,
            'audio_fft_length': 2048,
            'audio_num_mel_bins': 80,
            'audio_griffin_lim_num_iters': 50,
            'audio_griffin_lim_pre_amp': 1.2,
            'audio_max_db': 100,
            'audio_ref_db': 20,
        }
        sample_rate = AUDIO_PARAMS['audio_sample_rate']
        sr_kHz = sample_rate / 1000
        hop_length = int(AUDIO_PARAMS['audio_frame_step'] * sr_kHz)
        win_length = int(AUDIO_PARAMS['audio_frame_length'] * sr_kHz)

        spectrogram_features = librosa.stft(signal,
                                            AUDIO_PARAMS['audio_fft_length'],
                                            hop_length,
                                            win_length)
        spectrogram_features = np.abs(spectrogram_features)  # (1+n_fft//2, T)
        spectrogram_features = spectrogram_features.transpose()
        if not augmented:
            return spectrogram_features
        d_spectrogram_features = delta(spectrogram_features, 2)
        a_spectrogram_features = delta(d_spectrogram_features, 2)
        concatenated_features = np.concatenate((
                spectrogram_features,
                d_spectrogram_features,
                a_spectrogram_features
            ),
            axis=1
        )
        return concatenated_features

    @staticmethod
    def features_from_name(name, signal, rate=default_rate, filters_number=default_filters_number, augmented=default_augmented):
        return getattr(SpeechFeatures, name)(signal, rate, filters_number, augmented)
