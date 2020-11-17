#!/usr/bin/python
"""This script copies audio files into target dir with proper names and creates
training and validation filelists for both Tacotron2 and FastPitch models.
"""

import sys
import argparse
from pathlib import Path
from shutil import copy2
from fastcore.all import L
from fastai.data.transforms import get_text_files
from fastai.data.transforms import RandomSplitter
from fastprogress.fastprogress import progress_bar

def label_func(fname):
    "Return path to audio file corresponding to text `fname`"
    return Path(fname).parent.parent/'audio'/f'{Path(fname).stem}.flac'


def create_filelist(filelist: Path, idxs: L, files: L) -> None:
    """Create a file with audio `filelist` from `idxs` of `files`
    and copy audio files with new names into `target_audios_path`.

    Example:
        audio/EHD_120768D_206.flac|это демонстрирует его отношение к рабству.
        audio/EHD_120770D_068.flac|а в «юнион пасифик» их не трогали.

    """

    target_audios_path = filelist.parent/'audio'

    with open(filelist, "w+") as fl:
        for idx in progress_bar(idxs):
            episode, number = files[idx].parents[3].name, int(files[idx].stem)
            filename = f'{episode}_{number:03}.flac'
            fl.write(f'audio/{filename}|'+(files[idx]).open().read())
            fl.write('\n')
            if not (target_audios_path/filename).exists():
                copy2(label_func(files[idx]), target_audios_path/filename)
        #print('\n')


def create_mel_filelist(filelist: Path, idxs: L, files: L):
    """Creates a mel_duration_pitch `filelist` from `idxs` of `files`.
    Example:
        mels/EHD_120768D_206.pt|durations/EHD_120768D_206.pt|pitch_char/EHD_120768D_206.pt|это демонстрирует его отношение к рабству.
        mels/EHD_120770D_068.pt|durations/EHD_120768D_206.pt|pitch_char/EHD_120768D_206.pt|а в «юнион пасифик» их не трогали.

    """
    with open(filelist, "w+") as fl:
        for idx in progress_bar(idxs):
            episode, number = files[idx].parents[3].name, int(files[idx].stem)
            filename = f'{episode}_{number:03}.pt'
            fl.write(f'mels/{filename}|durations/{filename}|pitch_char/{filename}|'+(files[idx]).open().read())
            fl.write('\n')
        #print('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--dataset-path', type=str, required=True, help="Path to dataset.")
    parser.add_argument('-e', '--export-path', type=str, default='./', 
                        help="File to export items to.")
    parser.add_argument('-f', '--folders', nargs='+', default=[], help="Only these folders.")
    parser.add_argument('-p', '--valid-pct', type=float, default=0.1, 
                        help="Part of the dataset to use for validation. Default: 0.1")
    parser.add_argument('-s', '--seed', type=int, default=0, help="Seed to use for random split.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Report more info.")

    try:
        args = parser.parse_args()    
    except OSError:
        parser.print_help()
        sys.exit(1)

    target_path = Path(args.export_path)
    target_path.mkdir(exist_ok=True) 
    target_audios_path = target_path/'audio'
    target_audios_path.mkdir(exist_ok=True)

    files = get_text_files(args.dataset_path, folders=args.folders)

    if len(files) == 0:
        print("Empty dataset?")
        sys.exit(1)

    rs = RandomSplitter(valid_pct=args.valid_pct, seed=args.seed)
    train_idxs, val_idxs = rs(files)

    if args.verbose: 
        print(f'Dataset length: {len(files)} files.')
        print(f'Training/Validation set length: {len(train_idxs)}/{len(val_idxs)} files.')

    create_filelist(target_path/'train_filelist.txt', train_idxs, files)
    if args.verbose: 
        print(f'\nFilelist created: {target_path/"train_filelist.txt"}')
    else: 
        print('\n')
    create_filelist(target_path/'valid_filelist.txt', val_idxs,   files)
    if args.verbose: 
        print(f'\nFilelist created: {target_path/"valid_filelist.txt"}')
    else: 
        print('\n')        

    create_mel_filelist(target_path/'mel_dur_pitch_train_filelist.txt', train_idxs, files)
    if args.verbose: 
        print(f'\nFilelist created: {target_path/"mel_dur_pitch_train_filelist.txt"}')
    else: 
        print('\n')
    create_mel_filelist(target_path/'mel_dur_pitch_valid_filelist.txt', val_idxs,   files)
    if args.verbose: 
        print(f'\nFilelist created: {target_path/"mel_dur_pitch_valid_filelist.txt"}')
    else: 
        print('\n')


if __name__ == '__main__':
    main()
