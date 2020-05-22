""" Text Normalization Utility for Text-To-Speech dataset preparation
    by K.Dorichev github: @kdorichev
    May 2020
"""

import argparse
import nltk
from text_norm import *

def main():
    parser = argparse.ArgumentParser(description="Text Normalization Utility")
    parser.add_argument('infile')
    parser.add_argument('outfile')
    ars = parser.parse_args()
    
    sentences = []
    inf = open(ars.infile, 'r')
    of  = open(ars.outfile,'w')

    for line in inf:
        line = russian_cleaner(line)
        if line and ord(line[0]) == 65279: # BOM ZERO WIDTH NO-BREAK SPACE' (U+FEFF) (#65279) 
            line = line[1:] 
        
        if line is not '':
            sentences = (nltk.sent_tokenize(line, language="russian"))
            for s in sentences:
                if s.strip() is not '':
                    of.write(s)
                    of.write('\n')

    inf.close()
    of.close()

if __name__ == '__main__':
    main()
