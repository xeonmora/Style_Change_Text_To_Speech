import sys
import numpy as np
import pandas as pd
import torch
from scipy.io.wavfile import write

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
    df = pd.read_csv('story.csv')

    for count, value in enumerate(list(df['Sentence'])):
        print(count, value)
        mel = textToMel(value)
        audio = melToAudio(mel)
        write(f'./AudioClips_Netural/audio_{count}.wav', hparams.sampling_rate, audio)
