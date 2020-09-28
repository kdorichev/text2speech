# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_data.ipynb (unless otherwise specified).

__all__ = ['get_files', 'get_audio_files', 'get_txt_files', 'words_count', 'label_func', 'ReadTxt', 'word_in_files',
           'regex_files', 'accent_files', 'abbrev_files', 'unusual_files', 'plot_durations', 'drop_outliers',
           'create_filelist', 'create_mel_filelist']

# Cell
from pathlib import Path
from shutil import copy2

# Cell
import seaborn as sns
sns.set()

# Cell
from fastcore.all import *
from fastai.data.all import *
from fastaudio.core.all import *

# Cell
from spacy.lang.ru import Russian
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS

# Cell
from .text_norm import *

# Cell
def _get_files(p, fs, extensions=None):
    "Construct a list of `Path`s from a list of files `fs` in directory`p`"
    p = Path(p)
    res = [p/f for f in fs
           if not f.startswith('.') # not hidden file
              and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res

# Cell
def get_files(path, extensions=None, recurse=True, folders=None, followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    path = Path(path)
    folders = setify(folders)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)): # returns (dirpath, dirnames, filenames)
            if folders and not set(folders).issubset(p.split('/')): continue
            else: res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return L(res)

# Cell
def get_audio_files(path, recurse=True, folders=None):
    "Get audio files in `path` recursively, only in `folders`, if specified."
    return get_files(path, extensions=audio_extensions, recurse=recurse, folders=folders)

# Cell
def get_txt_files(path, recurse=True, folders=None):
    "Get text files in `path` recursively, only in `folders`, if specified."
    return get_files(path, extensions='.txt', recurse=recurse, folders=folders)

# Cell
def words_count(files, sort=True, drop_punkt=True):
    "Return a dict of words in `files` with counts, sorted by default"
    wc = {}
    nlp = Russian()
    russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)
    nlp.add_pipe(russian_tokenizer, name='russian_tokenizer')
    _punct = ".,?!;:+-«»–"

    for file in files:
        with open(file) as f:
            text = f.read()

        for token  in nlp((russian_cleaner(text))):
            if drop_punkt and (token.is_punct | token.is_space | (token.text in _punct + "». », !», ?» »:")): continue
            if token.text not in wc:
                wc[token.text] = 1
            else:
                wc[token.text] += 1

    if sort: return {k: v for k, v in sorted(wc.items(), key=lambda item: item[1], reverse=True)}
    else:    return wc

# Cell
def label_func(fname):
    "Return path to audio file corresponding to text `fname`"
    return Path(fname).parent.parent/'audio'/f'{Path(fname).stem}.flac'

# Cell
def ReadTxt(fn):
    "Read text from `fn`"
    fn = Path(fn) if isinstance(fn,str) else fn
    return fn.read().strip()

# Cell
def word_in_files(word, files, show=False, play=False):
    "Return an L list of `files` where `word` is present; optionally `show` and/or `play`"
    if not isinstance(files, L): files = L(files)

    nlp = Russian()
    russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)
    found_in = set()

    for f in files:
        doc = nlp(ReadTxt(f))
        for token in doc:
            if token.text == word:
                found_in.add(f)

    for item in found_in:
        if show: print(item); print(ReadTxt(item))
        if play: audio = AudioTensor.create(label_func(item)); audio.hear()
    return L(found_in)

# Cell
def regex_files(regexp, files, show=False, play=False):
    """Return an L list of text `files` where regular expression `regexp` is found.
    Optionally `show` text and/or `play` corresponding audio."""
    if not isinstance(files, L): files = L(files)
    found_in = set()

    for f in files:
        text = ReadTxt(f)
        if len(re.findall(regexp, text)) == 0: continue
        found_in.add(f)
        if show: print(text); print(f)
        if play: audio = AudioTensor.create(label_func(f)); audio.hear()
    return L(found_in)

# Cell
def accent_files(files=None, show=False, play=False):
    "Create a function that returns a `L` list of files where accent is marked with + in any word."
    def _inner(files, show=show, play=play):
        return regex_files(r"\b\w{1,}\+\w{0,}", files, show=show, play=play)
    return _inner

# Cell
def abbrev_files(files=None, show=False, play=False):
    "Create a function that returns a `L` list of files where abbreviated word is wrapped in *"
    def _inner(files, show=show, play=play):
        return regex_files(r"\*\w{1,}\*", files, show=show, play=play)
    return _inner

# Cell
def unusual_files(files=None, show=False, play=False):
    "Create a function that returns a `L` list of files where unusually pronounsed word is wrapped in _"
    def _inner(files, show=show, play=play):
        return regex_files(r"\b\_\w{1,}\_\b", files, show=show, play=play)
    return _inner

# Cell
def plot_durations(files, figsize=None, title: str = "Dataset Clip Duration Distribution"):
    "Plot audio `files` duration distribution"
    sum_dur = 0
    durations = []

    for f in files:
        at = AudioTensor.create(label_func(f))
        sum_dur += at.duration
        durations.append(at.duration)

    durations = torch.tensor(durations)
    figsize=(14,4) if figsize is None else figsize
    ax=plt.subplots(1,1,figsize=figsize)[1]

    max_dur = math.ceil(durations.max())
    sns.distplot(durations,rug=True,axlabel='sec',ax=ax)
    ttl = f"""{title}\n\
        {len(durations)} clips, {sum_dur/3600:.1f} hours\n\
        Min = {durations.min().item():.3f}\
        Mean = {durations.mean().item():.3f}\
        Max = {durations.max().item():.3f}
    """
    ax.set_title(ttl);
    ax.xaxis.set_ticks(range(0, max_dur+1));

# Cell
def drop_outliers(files, mindur=None, maxdur=None):
    "Return only those `files` with duration in (`mindur`,`maxdur`), drop outliers."
    if mindur is None and maxdur is None: return L(files)
    newfiles = []
    mindur = mindur if mindur is not None else 0
    maxdur = maxdur if maxdur is not None else 10000
    for f in files:
        at = AudioTensor.create(label_func(f))
        if mindur <= at.duration and at.duration <= maxdur:
            newfiles.append(f)
    return L(newfiles)

# Cell
def create_filelist(filelist: Path, idxs: L, files: L) -> None:
    """Create a file with audio `filelist` from `idxs` of `files`
    and copy audio files with new names into `target_audios_path`.

    Example:
        audio/EHD_120768D_206.flac|это демонстрирует его отношение к рабству.
        audio/EHD_120770D_068.flac|а в «юнион пасифик» их не трогали.

    """

    target_audios_path = filelist.parent/'audio'

    with open(filelist, "w+") as fl:
        for idx in idxs:
            episode, number = files[idx].parents[3].name, int(files[idx].stem)
            filename = f'{episode}_{number:03}.flac'
            fl.write(f'audio/{filename}|'+ReadTxt(files[idx]))
            fl.write('\n')
            if not (target_audios_path/filename).exists():
                copy2(label_func(files[idx]), target_audios_path/filename)

# Cell
def create_mel_filelist(filelist: Path, idxs: L, files: L):
    """Creates a mel_duration_pitch `filelist` from `idxs` of `files`.

    Example:
        mels/EHD_120768D_206.pt|durations/EHD_120768D_206.pt|pitch_char/EHD_120768D_206.pt|это демонстрирует его отношение к рабству.
        mels/EHD_120770D_068.pt|durations/EHD_120768D_206.pt|pitch_char/EHD_120768D_206.pt|а в «юнион пасифик» их не трогали.

    """

    with open(filelist, "w+") as fl:
        for idx in idxs:
            episode, number = files[idx].parents[3].name, int(files[idx].stem)
            filename = f'{episode}_{number:03}.pt'
            fl.write(f'mels/{filename}|durations/{filename}|pitch_char/{filename}|'+ReadTxt(files[idx]))
            fl.write('\n')