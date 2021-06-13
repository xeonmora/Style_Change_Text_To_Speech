import os
import shutil
import sys

sys.path.append('Melspectrogram_To_Audio/')
import numpy as np
import pandas as pd
import torch
import librosa
from scipy.io.wavfile import write
from pydub import AudioSegment

from Style_Change.model import load_model
from Style_Change.layers import TacotronSTFT
from Style_Change.data_utils import TextMelLoader, TextMelCollate
from Style_Change.text import cmudict, text_to_sequence
from Style_Change.hparams import create_hparams
from Melspectrogram_To_Audio.denoiser import Denoiser


def panner(signal, angle):
    angle = np.radians(angle)
    left = np.sqrt(2) / 2.0 * (np.cos(angle) - np.sin(angle)) * signal
    right = np.sqrt(2) / 2.0 * (np.cos(angle) + np.sin(angle)) * signal
    return np.dstack((left, right))[0]


def load_mel(path):
    audio, sampling_rate = librosa.core.load(path, sr=hparams.sampling_rate)
    audio = torch.from_numpy(audio)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.cuda()
    return melspec


hparams = create_hparams()

stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax)

checkpoint_path = "Style_Change/weight/checkpoint_37000"
mellotron = load_model(hparams).cuda().eval()
mellotron.load_state_dict(torch.load(checkpoint_path)['state_dict'])

waveglow_path = "Melspectrogram_To_Audio/weight/waveglow_256channels_universal_v5.pt"
waveglow = torch.load(waveglow_path)['model'].cuda().eval()
denoiser = Denoiser(waveglow).cuda().eval()

arpabet_dict = cmudict.CMUDict('Style_Change/data/cmu_dictionary')
audio_paths = 'Style_Change/data/examples_filelist.txt'


def styleChange(token):
    dataloader = TextMelLoader(audio_paths, hparams)
    datacollate = TextMelCollate(1)

    file_idx = 0
    audio_path, text, sid = dataloader.audiopaths_and_text[file_idx]

    # get audio path, encoded text, pitch contour and mel for gst
    text_encoded = torch.LongTensor(text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None, :].cuda()
    pitch_contour = dataloader[file_idx][3][None].cuda()
    mel = load_mel(audio_path)

    # load source data to obtain rhythm using tacotron 2 as a forced aligner
    x, y = mellotron.parse_batch(datacollate([dataloader[file_idx]]))

    with torch.no_grad():
        # get rhythm (alignment map) using tacotron 2
        mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = mellotron.forward(x)
        rhythm = rhythm.permute(1, 0, 2)

    speaker_id = torch.LongTensor([token]).cuda()

    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, gate_outputs, _ = mellotron.inference_noattention(
            (text_encoded, mel, speaker_id, pitch_contour, rhythm))

    mel = torch.autograd.Variable(mel_outputs_postnet.cuda())

    return mel


def melToAudio(mel):
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=0.666).float()

    audio = audio.cpu().numpy()[0]
    audio = audio / np.abs(audio).max()

    return audio


if __name__ == '__main__':
    if not os.path.isdir('./AudioClips_Emotion/'):
        os.mkdir('./AudioClips_Emotion/')
    else:
        shutil.rmtree('./AudioClips_Emotion/')
        os.mkdir('./AudioClips_Emotion/')

    df = pd.read_csv('story.csv')
    emotions = list(df['Emotion'])

    for count, value in enumerate(list(df['Sentence'])):
        print(count, value)

        file = open(r"Style_Change/data/examples_filelist.txt", "w")
        file.writelines(f"AudioClips_Netural/audio_{count}.wav|{value}|0")
        file.close()

        mel = styleChange(emotions[count])
        audio = melToAudio(mel)
        write(f'./AudioClips_Emotion/audio_{count}.wav', hparams.sampling_rate, audio)

    path = []
    for i in os.listdir('./AudioClips_Emotion'):
        path.append(i.split("_")[1].split(".")[0])

    list1 = [int(x) for x in path]
    list1.sort()

    path_list = []
    for i in list1:
        path_list.append(f"./AudioClips_Emotion/audio_{i}.wav")

    print(path_list)

    one_sec_segment = AudioSegment.silent(duration=1500)  # duration in milliseconds
    one_sec_segment.export('./AudioClips_Emotion/interval.wav', format="wav")

    sound = AudioSegment.from_wav("./AudioClips_Emotion/interval.wav")

    for i in path_list:
        sound = sound + AudioSegment.from_wav(i) + AudioSegment.from_wav("./AudioClips_Emotion/interval.wav")

    sound.export("Audio_Emotion.wav", format="wav")
