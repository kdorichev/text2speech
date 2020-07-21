# Adapted from
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch

# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
    Miscelaneous utility functions.
"""

__all__ = ['mask_from_lens', 'load_wav_to_torch', 'load_filepaths_and_text',
           'stats_filename', 'to_gpu', 'to_device_async', 'to_numpy']

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from librosa.core import load


def mask_from_lens(lens, max_len: Optional[int] = None):
    """Return an element-wise boolean mask of length less that `max_len`.

    Args:
        lens ([type]): [description]
        max_len (Optional[int]): max length. Defaults to None.

    Returns:
        tensor:
    """

    if max_len is None:
        max_len = int(lens.max().item())
    ids = torch.arange(max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


def load_wav_to_torch(full_path: str, sr: Optional[int] = 22050) -> Tuple[torch.Tensor, int]:
    """Load audio file from `full_path` with optional resamplling to `sr`.

    Args:
        full_path (str): path to audio file.
        sr (int, optional): sample rate to resample to.

    Returns:
        (torch.Tensor, sampling_rate)
    """

    data, sampling_rate = load(full_path, sr)
    return torch.from_numpy(data), sampling_rate

def _split_line(root, line, split="|") -> Tuple[str, str]:
    """Split a line from the filelists/*_filelist.txt files into a tuple consisting
    the `paths` prepended with the root of the dataset -- all the fields but last,
    and `text` -- the last field.

    Examples:
    wavs/LJ037-0219.wav|Oswald's Jacket

    mels/LJ030-0105.pt|durations/LJ030-0105.pt|pitch_char/LJ030-0105.pt|Communications in the motorcade.

    Args:
        root (str): path to the root directory of the dataset to prefix the paths fields.
        line (str): A line from the filelists/*.filelist.txt to split.
        split (str): A character separatr to split by. Defaults to '|'.

    Returns:
        tuple: A line split into a tuple 
            Examples:
                    ('LJSpeech/wavs/LJ037-0219.wav', "Oswald's Jacket")

                    ('LJSpeech/mels/LJ030-0105.pt',
                     'LJSpeech/durations/LJ030-0105.pt',
                     'LJSpeech/pitch_char/LJ030-0105.pt',
                     'Communications in the motorcade.')

    """

    parts = line.strip().split(split)
    paths, text = parts[:-1], parts[-1]
    return tuple(os.path.join(root, p) for p in paths) + (text,)


def load_filepaths_and_text(dataset_path, filename, split="|") -> list:
    """Parse the `filename` and return a list of tuples consisting the paths 
    to the saved tensors and the text.

    Args:
        dataset_path (str): path to the root directory of the dataset.
        filename (str): A file from the filelists/* to parse.
        split (str, optional): Felds separator. Defaults to "|".

    Returns:
        a list of tuples
        Example:
            [('LJSpeech/wavs/LJ037-0219.wav', "Oswald's Jacket")
                ...
             ('LJSpeech/wavs/LJ030-0109.wav', "The Vice-Presidential car")]
    """

    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [_split_line(dataset_path, line) for line in f]
    return filepaths_and_text


def stats_filename(dataset_path: str, filelist_path: str, feature_name: str) -> Path:
    """Construct filename to write the mean and std of the `feature_name`
    for the `filelist_path` part of the `dataset_path`.

    Args:
        dataset_path (str): Root path for the dataset.
        filelist_path (str): File from the filelists directory.
        feature_name (str): Name of the feature to include into the filename.
            Example: pitch_char

    Returns:
        Path: [description]
    """

    stem = Path(filelist_path).stem
    return Path(dataset_path, f'{feature_name}_stats__{stem}.json')


def to_gpu(x):
    """Move `x` to cuda device.

    Args:
        x (tensor)

    Returns:
        torch.Variable
    """

    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def to_device_async(tensor, device):
    return tensor.to(device, non_blocking=True)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Cast tensor `x` to numpy.ndarray

    Args:
        x ([type])

    Returns:
        numpy.ndarray
    """

    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x
