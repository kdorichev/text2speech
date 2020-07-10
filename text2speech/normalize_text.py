""" Text Normalization Utility for Text-To-Speech dataset preparation
    by K.Dorichev github: @kdorichev
    May 2020
"""

import argparse
import nltk
from text_norm import *

__version__ = '0.3'
__date__ = '10.07.2020'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('--purgedots', action='store_true',
                        help='If set, `...`, `…` will be purged. Else replaced with `.`')
    parser.add_argument('--version', action='version',
                    version=f'%(prog)s, version {__version__}, {__date__}', help='Print script version.')
    
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
