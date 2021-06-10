import sys
import numpy as np
import torch
from scipy.io.wavfile import write
from Text_To_Melspectrogram.hparams import create_hparams
from Text_To_Melspectrogram.text import text_to_sequence
from Text_To_Melspectrogram.model import Tacotron2

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


# Single sentence text to audio
def singleTextExample(text):
    MAX_WAV_VALUE = 32768.0
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    mel = torch.autograd.Variable(mel_outputs_postnet.cuda())

    with torch.no_grad():
        audio = MAX_WAV_VALUE * waveglow.infer(mel, sigma=0.666)[0]

    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    write('audio.wav', hparams.sampling_rate, audio)


def multiLineExample():
    final_audio = np.array(81408, )
    spectogram = []

    with open('story.txt') as f:
        story_text = f.readlines()

    print("mel spectrogram making....")
    for i in story_text:
        sequence = np.array(text_to_sequence(i[:-1], ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()

        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        print(mel_outputs_postnet.dtype)
        spectogram.append(mel_outputs_postnet)

    print("audio making....")
    with torch.no_grad():
        for i in spectogram:
            audio = waveglow.infer(i, sigma=0.666).float()
            audio = audio.cpu().numpy()[0]
            # audio = audio / np.abs(audio).max()
            final_audio = np.append(final_audio, audio)

    write('complete_audio.wav', hparams.sampling_rate, final_audio)


if __name__ == '__main__':
    singleTextExample("The Donkey he came to a kingdom where he heard there was an old king with a wonderfully beautiful daughter.")
    # multiLineExample()
