#!/usr/bin/env python3
"""This script copies audio files into target dir with proper names and creates 
training and validation filelists.
"""

import argparse
from pathlib import Path
#from shutil import copy2

from fastcore.all import L
from text2speech.data import * #get_txt_files, create_filelist, create_mel_filelist
from fastai.data.transforms import RandomSplitter


def main():
    """Check full or up to `number_items` of `dataset_path` or a single `input_file`
    using VOSK speech recognition and report inconsistencies with text file(s).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--dataset-path', type=str, required=True, help="Path to dataset.")
    parser.add_argument('-e', '--export-path', type=str, default='./', 
                        help="File to export NOK items to.")
    parser.add_argument('-f', '--folders', nargs='+', default=[], help="Only these folders.")
    parser.add_argument('-p', '--valid-pct', type=float, default=0.1, 
                        help="Part of the dataset to use for validation. Default: 0.1")
    parser.add_argument('-s', '--seed', type=int, default=0, help="Seed to use for random split.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Report more info.")

    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)

    target_path = Path(args.export_path)
    target_path.mkdir(exist_ok=True) 
    target_audios_path = target_path/'audio'
    target_audios_path.mkdir(exist_ok=True)
    
    files = get_txt_files(args.dataset_path, folders=args.folders)
    if args.verbose: 
        print(f'Dataset length: {len(files)} files.')

    rs = RandomSplitter(valid_pct=args.valid_pct, seed=args.seed)
    train_idxs, val_idxs = rs(files)

    if args.verbose: 
        print(f'Training/Validation set length: {len(train_idxs)}/{len(val_idxs)} files.')
    
    create_filelist(target_path/'train_filelist.txt', train_idxs, files)
    if args.verbose: print(f'Filelist created: {target_path/"train_filelist.txt"}')
    create_filelist(target_path/'valid_filelist.txt', val_idxs,   files)
    if args.verbose: print(f'Filelist created: {target_path/"valid_filelist.txt"}')
    
    create_mel_filelist(target_path/'mel_dur_pitch_train_filelist.txt', train_idxs, files)
    if args.verbose: print(f'Filelist created: {target_path/"mel_dur_pitch_train_filelist.txt"}')
    create_mel_filelist(target_path/'mel_dur_pitch_valid_filelist.txt', val_idxs,   files)
    if args.verbose: print(f'Filelist created: {target_path/"mel_dur_pitch_valid_filelist.txt"}')

if __name__ == '__main__':
    main()
