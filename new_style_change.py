# Import modules
import sys
import numpy as np
import torch
import librosa
from scipy.io.wavfile import write

from Text_To_Melspectrogram.hparams import create_hparams as hparams1
from Text_To_Melspectrogram.text import text_to_sequence
from Text_To_Melspectrogram.model import Tacotron2

from Style_Change.hparams import create_hparams as hparams2
from Style_Change.model import load_model
from Style_Change.layers import TacotronSTFT
from Style_Change.data_utils import TextMelLoader, TextMelCollate
from Style_Change.text import cmudict, text_to_sequence

sys.path.append('Melspectrogram_To_Audio/')
from denoiser import Denoiser

hparams1 = hparams1()
hparams2 = hparams2()

# Load text-to-mel model
checkpoint_path = "./Text_To_Melspectrogram/weight/tacotron2_statedict.pt"
model = Tacotron2(hparams1).cuda()
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

# Load style-change model
stft = TacotronSTFT(hparams2.filter_length, hparams2.hop_length, hparams2.win_length,
                    hparams2.n_mel_channels, hparams2.sampling_rate, hparams2.mel_fmin,
                    hparams2.mel_fmax)

checkpoint_path = "./Style_Change/weight/checkpoint_18000"
mellotron = load_model(hparams2).cuda().eval()
mellotron.load_state_dict(torch.load(checkpoint_path)['state_dict'])

# Loade mel-to-audio model
waveglow_path = "./Melspectrogram_To_Audio/weight/waveglow_256channels_universal_v5.pt"
waveglow = torch.load(waveglow_path)['model'].cuda().eval()
denoiser = Denoiser(waveglow).cuda().eval()


# Single sentence to melspectogram
def textToMel(text):
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    mel = torch.autograd.Variable(mel_outputs_postnet.cuda())

    return mel


# Style change of the mel spectogram
def StyleChange(mel, token):
    pass


# Generate audio of the that melspectogram
def melToAudio(mel):
    MAX_WAV_VALUE = 32768.0
    with torch.no_grad():
        audio = MAX_WAV_VALUE * waveglow.infer(mel, sigma=0.666)[0]

    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    write('./Audio_files/audio.wav', hparams1.sampling_rate, audio)


# main run
if __name__ == '__main___':
    mel = textToMel(
        "The Donkey he came to a kingdom where he heard there was an old king with a wonderfully beautiful daughter.")
