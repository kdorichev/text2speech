# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_text_norm.ipynb (unless otherwise specified).

__all__ = ['collapse_whitespace', 'lowercase', 'check_no_numbers', 'remove_specials', 'expand_abbreviations',
           'unify_dash_hyphen', 'rm_quot_marks', 'basic_cleaner', 'russian_cleaner']

# Cell
import re

# Cell
def collapse_whitespace(text: str) -> str:
    "Replace multiple various whitespaces with a single space, strip leading and trailing spaces."

    return re.sub(r'[\s\ufeff\u200b\u2060]+', ' ', text).strip()

# Cell
def lowercase(text: str) -> str:
    "Convert `text` to lower case."

    return text.lower()

# Cell
def check_no_numbers(text: str) -> list:
    "Return a list of digits, or empty list, if not found."

    return re.findall(r"(\d+)", text)

# Internal Cell
_specials = [(re.compile(f'{x[0]}'), x[1]) for x in [
    ('\(?\d\d[:.]\d\d\)?', ''),  # timestamps
    ('!\.{1,}', '!'), # !. -> !
    ('\?\.{1,}', '?'),# ?. -> ?
    ('\/', ''),
    ('\s(…)', ' '),
    ('…', '.'),
    ('\.{3}', '.'),  # ... -> .
    ('\.{2}', ''),   # pause .. removed
]]

# Cell
def remove_specials(text: str, purge_digits: bool=None) -> str:
    "Replace predefined in `_specials` sequence of characters"

    for regex, replacement in _specials:
        text = re.sub(regex, replacement, text)
    if purge_digits:
        text = re.sub('\d', '', text)
    return text

# Cell
_abbreviations = [(re.compile(f'\\b{x[0]}', re.IGNORECASE), x[1]) for x in [
  ('т\.е\.', 'то есть'),
  ('т\.к\.', 'так как'),
  ('и т\.д\.', 'и так далее.'),
  ('и т\.п\.', 'и тому подобное.')
]]

# Cell
def expand_abbreviations(text: str) -> str:
    "`expand_abbreviations()` defined in `_abbreviations`"

    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

# Cell
def unify_dash_hyphen(text: str) -> str:
    "Unify dash and hyphen symbols -- replace with emdash or hyphen, separate with space."

    text = re.sub('[\u2212\u2012\u2014]', '\u2013', text) # replace minus sign, figure dash, em dash with en dash
    text = re.sub('[\u2010\u2011]', '\u002d', text)  # hyphen, non-breaking hyphen
    text = re.sub('\s*?(\u2013)\s*?',' \g<1> ',text)
    return text

# Cell
def rm_quot_marks(text: str) -> str:
    """Remove quotation marks from `text`."""
    # \u0022\u0027\u00ab\u00bb\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f\u2039\u203a\u276e\u276f\u275b\u275c\u275d\u275e\u275f\u2760\u2e42\u301d\u301e\u301f
    return re.sub('["\'«»‘’‚‛“”„‟‹›❮❯❛❜❝❞❟❠]','',text)

# Cell
def basic_cleaner(text: str) -> str:
    "Basic pipeline: lowercase and collapse whitespaces."
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

# Cell
def russian_cleaner(text, purge_digits=True):
    "Pipeline for Russian text cleaning: lowercase, expand_abbreviations, remove_specials, collapse_whitespace."
    text = lowercase(text)
    text = expand_abbreviations(text)
    text = remove_specials(text, purge_digits=purge_digits)
    text = unify_dash_hyphen(text)
    text = rm_quot_marks(text)
    text = collapse_whitespace(text)
    return text