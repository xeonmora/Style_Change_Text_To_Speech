import os
import shutil
import sys
import numpy as np
import pandas as pd
import torch
from scipy.io.wavfile import write
from pydub import AudioSegment

from Text_To_Melspectrogram.hparams import create_hparams
from Text_To_Melspectrogram.model import Tacotron2
from Text_To_Melspectrogram.text import text_to_sequence

sys.path.append('Melspectrogram_To_Audio/')
from denoiser import Denoiser

hparams = create_hparams()
hparams.sampling_rate = 22050

checkpoint_path = "./Text_To_Melspectrogram/weight/checkpoint_50000.pt"
model = Tacotron2(hparams).cuda()
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

waveglow_path = "Melspectrogram_To_Audio/weight/waveglow_256channels_universal_v5.pt"
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()

for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)


def textToMel(text):
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    mel = torch.autograd.Variable(mel_outputs_postnet.cuda())

    return mel


def melToAudio(mel):
    MAX_WAV_VALUE = 32768.0
    with torch.no_grad():
        audio = MAX_WAV_VALUE * waveglow.infer(mel, sigma=0.666)[0]

    audio = audio.cpu().numpy()
    audio = audio.astype('int16')

    return audio


if __name__ == '__main__':
    if not os.path.isdir('./AudioClips_Netural/'):
        os.mkdir('./AudioClips_Netural/')
    else:
        shutil.rmtree('./AudioClips_Netural/')
        os.mkdir('./AudioClips_Netural/')

    df = pd.read_csv('story.csv')

    for count, value in enumerate(list(df['Sentence'])):
        print(count, value)
        mel = textToMel(value)
        audio = melToAudio(mel)
        write(f'./AudioClips_Netural/audio_{count}.wav', hparams.sampling_rate, audio)

    path = []
    for i in os.listdir('./AudioClips_Netural'):
        path.append(i.split("_")[1].split(".")[0])

    list1 = [int(x) for x in path]
    list1.sort()

    path_list = []
    for i in list1:
        path_list.append(f"./AudioClips_Netural/audio_{i}.wav")

    print(path_list)

    one_sec_segment = AudioSegment.silent(duration=1500)  # duration in milliseconds
    one_sec_segment.export('./AudioClips_Netural/interval.wav', format="wav")

    sound = AudioSegment.from_wav("./AudioClips_Netural/interval.wav")

    for i in path_list:
        sound = sound + AudioSegment.from_wav(i) + AudioSegment.from_wav("./AudioClips_Netural/interval.wav")

    sound.export("Audio.wav", format="wav")