""" Text Normalization Utility for Text-To-Speech dataset preparation
    by K.Dorichev github: @kdorichev
    Sept 2020
"""

import argparse
from text_norm import *
from razdel import sentenize

__version__ = '0.4'
__date__ = '04.09.2020'

def main():
    """Split `infile` into sentences and clean the text.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('--purgedots', action='store_true',
                        help='If set, `...`, `…` will be purged. Else replaced with `.`')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s, version {__version__}, {__date__}',
                        help='Print script version.')

    args = parser.parse_args()

    sentences = []

    try:
        with open(args.infile, 'r') as inf:
            try:
                with open(args.outfile, 'w') as of:

                    for line in inf:
                        line = russian_cleaner(line, _purge_dots=args.purgedots)

                        if line != '':
                            sentences = sentenize(line)
                            for s in sentences:
                                if s != '':
                                    of.write(s.text)
                                    of.write("\n")
            except IOError:
                print(f"Could not create output file: {args.outfile}")
                return          

    except IOError:
        print(f"Could not read file: {args.infile}")


if __name__ == '__main__':
    main()
