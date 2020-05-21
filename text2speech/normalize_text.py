""" Text Normalization Utility for Text-To-Speech dataset preparation
    by K.Dorichev github: @kdorichev
    May 2020
"""
import sys
import re
import argparse
import fileinput
import nltk
from text_norm import *

def main():
    parser = argparse.ArgumentParser(description="Text Normalization Utility")
    parser.add_argument('infile', nargs='?',  type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)    
#    parser.add_argument("-i", "--input", required=True, help="Input text for normalization")
#    parser.add_argument("-o", "--output", required=False, help="Output file name.")
    ars = parser.parse_args()

    text = []
    #with fileinput
    for line in fileinput.input(ars.infile):
        print(russian_cleaner(line))
        text.append(line)
    
    text = nltk.sent_tokenize(text, language="russian")

    with open(ars.outfile) as of:
        of.write(text)
            


if __name__ == '__main__':
    main()