#!/usr/bin/env python3
#"""Dataset integrity check utility."""

import sys
import os
import re
import wave
import json
import subprocess
import argparse
from typing import Tuple

from vosk import Model, KaldiRecognizer, SetLogLevel
from pathlib import Path
from text2speech.data import *
from text2speech.text_norm import *
from fastcore.all import L


parser = argparse.ArgumentParser(description = __doc__)
parser.add_argument('-d','--dataset-path', type=str, help = "Path to dataset.")
parser.add_argument('-i','--input-file', type=str, help = "Text file to parse.")
parser.add_argument('-e','--export-path', type=str, help = "File to export NOK items to.")
parser.add_argument('-f','--folders', nargs='+', default=[], help = "Only these folders.")
parser.add_argument('-n','--number-items', type=int, help = "Number of items to parse only.")
parser.add_argument('-v','--verbose', action='store_true', help = "Report more info.")

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
    
SetLogLevel(-1)
model = "/home/condor/git/vosk-api/vosk-model-ru-0.10/"
if not os.path.exists(model):
    print ("Please download the model from https://github.com/alphacep/vosk-api/blob/master/doc/models.md and unpack as 'model' in the current folder.")
    exit (1)

sample_rate=48000
model = Model(model)
rec = KaldiRecognizer(model, sample_rate)

if args.input_file is not None:
    files = L(Path(args.input_file))
elif args.dataset_path is not None:
    files = get_txt_files(args.dataset_path, folders=args.folders)
    assert len(files) > 0 #, f"Empty dataset? {args.dataset_path}"
else:
    parser.print_help()
    sys.exit(0)

if args.number_items is not None:
    files = files[:args.number_items]  

print(f'Parsing {len(files)} files from {args.dataset_path}')

files = files.sorted()

ok_count = 0
nok_files = dict()

for f in files:
    af = label_func(f)
    with subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i', af,
                           '-ar', str(sample_rate), '-ac', '1', '-f', 's16le', '-'],
                            stdout=subprocess.PIPE) as process:
        asr_text = ""
        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                if asr_text != "":
                    asr_text += " "
                asr_text += json.loads(rec.Result())["text"]

        if asr_text != "":
            asr_text += " "
        asr_text += json.loads(rec.FinalResult())["text"]

        orig_txt = ReadTxt(f)
        cln_txt = russian_cleaner(orig_txt)
        cln_txt = re.sub(r'[\.,?!*_:+]','', cln_txt)
        cln_txt = re.sub(r' [-–] ',' ', cln_txt)
        
        is_equal, cln_txt, asr_text = texts_equal(cln_txt, asr_text)
        if is_equal:
            ok_count += 1
        else:
            print(f'AUD: {af} - NOK')
            if args.verbose:
                print(f'TXT: {orig_txt}')
                print(f'CLN: {cln_txt}')
                print(f'ASR: {asr_text}\n')
            nok_files[int(f.stem)] = (f, asr_text)

print(f'NOK {(len(files) - ok_count) / len(files) *100:.2f}%')

export_file = args.export_path if args.export_path is not None else files[0].parents[3].name+'_NOK.log'

# nok_file_numbers = L()
# for f in nok_files:
#     nok_file_numbers += int(f.stem)

with open(export_file, "w+") as outfile:
    for key in sorted(nok_files.keys()):
        item = nok_files[key]
        outfile.write(f'{str(item[0].stem)}\t{item[1]}')
        outfile.write('\n')
