""" Text Normalization Utility for Text-To-Speech dataset preparation
    by K.Dorichev github: @kdorichev
    May 2020
"""

import argparse
import nltk
from text_norm import *

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="Text Normalization Utility")
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('--purgedots', type=str2bool, nargs='?', const=True, default=False,
                        help='If set, `...` will be purged. Else replaced with `.`')
    ars = parser.parse_args()

    sentences = []
    inf = open(ars.infile, 'r')
    of  = open(ars.outfile,'w')

    for line in inf:
        line = russian_cleaner(line, _purge_dots = ars.purgedots)
        
        if line != '':
            sentences = (nltk.sent_tokenize(line, language="russian"))
            for s in sentences:
                if s != '':
                    of.write(s)
                    of.write('\n')

    inf.close()
    of.close()

if __name__ == '__main__':
    main()
