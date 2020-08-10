# Adapted from
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch

# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
"""TTS Data Pre-processing."""

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import numpy as np
import parselmouth
import dllogger as DLLogger

from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from common import utils
from inference import load_and_setup_model
from tacotron2.data_function import TextMelLoader, TextMelCollate, batch_to_gpu


def parse_args(parser):
    """Parse commandline arguments."""

    parser.add_argument('--tacotron2-checkpoint', type=str,
                        help='full path to the generator checkpoint file')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    # Mel extraction
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('--wav-text-filelist', required=True,
                        type=str, help='Path to file with audio paths and text')
    parser.add_argument('--text-cleaners', nargs='*',
                        default=['russian_cleaner'], type=str,
                        help='Type of text cleaners for input text')
    parser.add_argument('--max-wav-value', default=1.0, type=float,
                        help='Maximum audiowave value')
    parser.add_argument('--sampling-rate', default=48000, type=int,
                        help='Sampling rate')
    parser.add_argument('--filter-length', default=1024, type=int,
                        help='Filter length')
    parser.add_argument('--hop-length', default=256, type=int,
                        help='Hop (stride) length')
    parser.add_argument('--win-length', default=1024, type=int,
                        help='Window length')
    parser.add_argument('--mel-fmin', default=0.0, type=float,
                        help='Minimum mel frequency')
    parser.add_argument('--mel-fmax', default=8000.0, type=float,
                        help='Maximum mel frequency')
    # Duration extraction
    parser.add_argument('--extract-mels', action='store_true',
                        help='Calculate spectrograms from .wav files')
    parser.add_argument('--extract-mels-teacher', action='store_true',
                        help='Extract Taco-generated mel-spectrograms for KD')
    parser.add_argument('--extract-durations', action='store_true',
                        help='Extract char durations from attention matrices')
    parser.add_argument('--extract-attentions', action='store_true',
                        help='Extract full attention matrices')
    parser.add_argument('--extract-pitch-mel', action='store_true',
                        help='Extract pitch')
    parser.add_argument('--extract-pitch-char', action='store_true',
                        help='Extract pitch averaged over input characters')
    parser.add_argument('--extract-pitch-trichar', action='store_true',
                        help='Extract pitch averaged over input characters')
    parser.add_argument('--train-mode', action='store_true',
                        help='Run the model in .train() mode')
    parser.add_argument('--cuda', action='store_true',
                        help='Extract mels on a GPU using CUDA')
    return parser


class FilenamedLoader(TextMelLoader):
    """A Loader that adds a filename to the text and mel."""

    def __init__(self, filenames: list, *args, **kwargs):
        super(FilenamedLoader, self).__init__(*args, **kwargs)
        self.filenames = filenames

    def __getitem__(self, index) ->Tuple[torch.IntTensor, torch.Tensor, int, Path]:
        """Adds Path of filename at the `index` element of the dataset.

        Args:
            index (int): index of file in `filenames`

        Returns:
            Tuple[torch.IntTensor, torch.Tensor, int, Path]: text, mel, text_len, filename
        """
        mel_text = super(FilenamedLoader, self).__getitem__(index)
        return mel_text + (self.filenames[index],)


def maybe_pad(vec: np.array, l: int):
    """Pad an numpy.array with `0` if less than 3 elements than `l`, or truncate.

    Args:
        vec (np.array): An array to pad or truncate
        l (int): required length

    Returns:
        np.array: Padded or truncated vector
    """

    assert np.abs(vec.shape[0] - l) <= 3
    vec = vec[:l]
    if vec.shape[0] < l:
        vec = np.pad(vec, pad_width=(0, l - vec.shape[0]))
    return vec


def dur_chunk_sizes(n, ary):
    """Split a single duration into almost-equally-sized chunks

    Examples:
      dur_chunk(3, 2) --> [2, 1]
      dur_chunk(3, 3) --> [1, 1, 1]
      dur_chunk(5, 3) --> [2, 2, 1]
    """

    ret = np.ones((ary,), dtype=np.int32) * (n // ary)
    ret[:n % ary] = n // ary + 1
    assert ret.sum() == n
    return ret


def calculate_pitch(audio_file: str, durs: np.array) -> Tuple[np.array, np.array, np.array]:
    """Calculate pitches of phonemes in `audio_file`.

    Args:
        audio_file (str): Audio file to read from.
        durs (np.array): Durations of phonemes.

    Returns:
        Tuple[np.array, np.array, np.array]: pitch_mel, pitch_char, pitch_trichar
    """
    mel_len = durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))
    snd = parselmouth.Sound(audio_file)
    pitch = snd.to_pitch(time_step=snd.duration / (mel_len + 3)
                         ).selected_array['frequency']
    assert np.abs(mel_len - pitch.shape[0]) <= 1.0

    # Average pitch over characters
    pitch_char = np.zeros((durs.shape[0],), dtype=np.float)
    for idx, a, b in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
        pitch_char[idx] = np.mean(values) if len(values) > 0 else 0.0

    # Average to three values per character
    pitch_trichar = np.zeros((3 * durs.shape[0],), dtype=np.float)

    durs_tri = np.concatenate([dur_chunk_sizes(d, 3) for d in durs])
    durs_tri_cum = np.cumsum(np.pad(durs_tri, (1, 0)))

    for idx, a, b in zip(range(3 * mel_len), durs_tri_cum[:-1], durs_tri_cum[1:]):
        values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
        pitch_trichar[idx] = np.mean(values) if len(values) > 0 else 0.0

    pitch_mel = maybe_pad(pitch, mel_len)
    pitch_char = maybe_pad(pitch_char, len(durs))
    pitch_trichar = maybe_pad(pitch_trichar, len(durs_tri))

    return pitch_mel, pitch_char, pitch_trichar


def normalize_pitch_vectors(pitch_vecs) -> Tuple[np.float64, np.float64]:
    """Normalize nonzero pitch vectors using calculated `mean` and `std`

    Args:
        pitch_vecs ([type]): [description]

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """

    nonzeros = np.concatenate([v[np.where(v != 0.0)[0]]
                               for v in pitch_vecs.values()])
    mean, std = np.mean(nonzeros), np.std(nonzeros)

    for v in pitch_vecs.values():
        zero_idxs = np.where(v == 0.0)[0]
        v -= mean
        v /= std
        v[zero_idxs] = 0.0

    return mean, std


def save_stats(dataset_path: str, wav_text_filelist: str, feature_name: str, mean: np.float64, std: np.float64) -> None:
    """Save `mean` and `std` of `wav_text_filelist` of `dataset_path` into a json-file.

    Args:
        dataset_path (str): Root path to the dataset
        wav_text_filelist (str): ljs_audio_text_*_filelist
        feature_name (str): Name of the feature to include into the filename.
            Example: pitch_char
        mean (np.float64): Mean
        std (np.float64): Standard deviation
    """

    fpath = utils.stats_filename(dataset_path, wav_text_filelist, feature_name)
    with open(fpath, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f, indent=4)


def main():
    """Main logic of the script.

    Raises:
        ValueError: In case of unknown arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    if args.extract_pitch_char:
        assert args.extract_durations, "Durations required for pitch extraction"

    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, args.log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k, v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})

    model = load_and_setup_model(
        'Tacotron2', parser, args.tacotron2_checkpoint, amp=False,
        device=torch.device('cuda' if args.cuda else 'cpu'),
        forward_is_infer=False, ema=False)

    if args.train_mode:
        model.train()

    # n_mel_channels arg has been consumed by model's arg parser
    args.n_mel_channels = model.n_mel_channels

    for datum in ('mels', 'mels_teacher', 'attentions', 'durations',
                  'pitch_mel', 'pitch_char', 'pitch_trichar'):
        if getattr(args, f'extract_{datum}'):
            Path(args.dataset_path, datum).mkdir(parents=False, exist_ok=True)

    filenames = [Path(l.split('|')[0])  # .stem
                 for l in open(args.wav_text_filelist, 'r')]
    dataset = FilenamedLoader(filenames, args.dataset_path, args.wav_text_filelist,
                              args, load_mel_from_disk=False)
    # TextMelCollate supports only n_frames_per_step=1
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             sampler=None, num_workers=0,
                             collate_fn=TextMelCollate(1),
                             pin_memory=False, drop_last=False)

    pitch_vecs = {'mel': {}, 'char': {}, 'trichar': {}}

    for i, batch in enumerate(data_loader):
        tik = time.time()
        fnames = batch[-1]
        x, _, _ = batch_to_gpu(batch[:-1])
        _, text_lens, mels_padded, _, mel_lens = x

        for j, mel in enumerate(mels_padded):
            fpath = Path(args.dataset_path)/'mels'/Path(Path(fnames[j]).name).with_suffix('.pt')
            torch.save(mel[:, :mel_lens[j]].cpu(), fpath)

        with torch.no_grad():
            out_mels, out_mels_postnet, _, alignments = model.forward(x)

        if args.extract_mels_teacher:
            for j, mel in enumerate(out_mels_postnet):
                fpath = Path(args.dataset_path)/'mels_teacher'/Path(Path(fnames[j]).name).with_suffix('.pt')
                torch.save(mel[:, :mel_lens[j]].cpu(), fpath)

        if args.extract_attentions:
            for j, ali in enumerate(alignments):
                ali = ali[:mel_lens[j],:text_lens[j]]
                fpath = Path(args.dataset_path)/'attentions'/Path(Path(fnames[j]).name).with_suffix('.pt')
                torch.save(ali.cpu(), fpath)

        durations = []
        if args.extract_durations:
            for j, ali in enumerate(alignments):
                text_len = text_lens[j]
                ali = ali[:mel_lens[j],:text_len]
                dur = torch.histc(torch.argmax(ali, dim=1), min=0,
                                  max=text_len-1, bins=text_len)
                durations.append(dur)
                fpath = Path(args.dataset_path)/'durations'/Path(Path(fnames[j]).name).with_suffix('.pt')
                torch.save(dur.cpu().int(), fpath)

        if args.extract_pitch_mel or args.extract_pitch_char or args.extract_pitch_trichar:
            for j, dur in enumerate(durations):
                # fpath = Path(args.dataset_path)/'pitch_char'/Path(Path(fnames[j]).name).with_suffix('.pt')
                fname = Path(fnames[j]).name
                audio_file = Path(args.dataset_path)/'audio'/Path(fname).with_suffix('.flac')
                # Path(args.dataset_path, 'wavs', fnames[j] + '.wav')
                p_mel, p_char, p_trichar = calculate_pitch(str(audio_file), dur.cpu().numpy())
                pitch_vecs['mel'][str(fname)] = p_mel
                pitch_vecs['char'][str(fname)] = p_char
                pitch_vecs['trichar'][str(fname)] = p_trichar

        nseconds = time.time() - tik
        DLLogger.log(step=f'Batch {i+1}/{len(data_loader)} ({nseconds:.2f}s)', data={})

    if args.extract_pitch_mel:
        normalize_pitch_vectors(pitch_vecs['mel'])
        for fname, pitch in pitch_vecs['mel'].items():
            fpath = Path(args.dataset_path)/'pitch_mel'/Path(fname).with_suffix('.pt')
            torch.save(torch.from_numpy(pitch), fpath)

    if args.extract_pitch_char:
        mean, std = normalize_pitch_vectors(pitch_vecs['char'])
        for fname, pitch in pitch_vecs['char'].items():
            fpath = Path(args.dataset_path)/'pitch_char'/Path(fname).with_suffix('.pt')
            torch.save(torch.from_numpy(pitch), fpath)
        save_stats(args.dataset_path, args.wav_text_filelist, 'pitch_char',
                   mean, std)

    if args.extract_pitch_trichar:
        normalize_pitch_vectors(pitch_vecs['trichar'])
        for fname, pitch in pitch_vecs['trichar'].items():
            fpath = Path(args.dataset_path)/'pitch_trichar'/Path(fname).with_suffix('.pt')
            torch.save(torch.from_numpy(pitch), fpath)

    DLLogger.flush()


if __name__ == '__main__':
    main()
