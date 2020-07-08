# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_transcribe.ipynb (unless otherwise specified).

__all__ = ['VoskTranscribe']

# Cell
class VoskTranscribe():
    """Transcribe an accented word using rules of `VOSK` library.

       For example:

       абстракциони+стов
       абстра+кцию
       абстра+кция

       абстракционистов a0 b s t r a0 k c i0 o0 nj i1 s t o0 v
       абстракцию a0 b s t r a1 k c i0 j u0
       абстракция a0 b s t r a1 k c i0 j a0

       The code is adapted from `vosk-model-ru-0.10/extra/scripts/dictionary.py` https://alphacephei.com/vosk/
    """

    def __init__(self, acc_before=False):
        """Create a `VoskTranscribe`r.

           Args:
            acc_before (bool): Accent marked with `+` before (True) or after (False) a vowel. Default=False.
                Example: "сл+ива" -- use `acc_before = True`, like in `VOSK`.
                         "сли+ва" -- use `acc_before = False`, like in `russian_g2p`.

        """
        self.acc_before = acc_before
        self.softletters=set("яёюиье")
        self.startsyl=set("#ъьаяоёуюэеиы-")
        self.others = set("#+-ьъ")
        self.softhard_cons = {
            "б" : "b",
            "в" : "v",
            "г" : "g",
            "Г" : "g",
            "д" : "d",
            "з" : "z",
            "к" : "k",
            "л" : "l",
            "м" : "m",
            "н" : "n",
            "п" : "p",
            "р" : "r",
            "с" : "s",
            "т" : "t",
            "ф" : "f",
            "х" : "h"
        }

        self.other_cons = {
            "ж" : "zh",
            "ц" : "c",
            "ч" : "ch",
            "ш" : "sh",
            "щ" : "sch",
            "й" : "j"
        }

        self.vowels = {
            "а" : "a",
            "я" : "a",
            "у" : "u",
            "ю" : "u",
            "о" : "o",
            "ё" : "o",
            "э" : "e",
            "е" : "e",
            "и" : "i",
            "ы" : "y",
        }


    def __call__(self, stressword: str) -> str:
        """To call class instance as a function."""
        return self.convert(stressword)


    def __pallatize(self, phones: list) -> list:
        """Transcribe consonant phones.

        Args:
            phones (list): tuples of phones marked: 0 -- not stressed, 1 -- stressed.
                Example: [('#', 0), ('с', 0), ('л', 0), ('и', 1), ('в', 0), ('а', 0), ('#', 0)]

        Returns:
            list of tuples: consonants transcribed.
                Example: [('#', 0), ('s', 0), ('lj', 0), ('и', 1), ('v', 0), ('а', 0), ('#', 0)]
        """

        for i, (ph, _) in enumerate(phones[:-1]):
            if ph in self.softhard_cons:
                if phones[i+1][0] in self.softletters:
                    phones[i] = (self.softhard_cons[ph] + "j", 0)
                else:
                    phones[i] = (self.softhard_cons[ph], 0)
            if ph in self.other_cons:
                phones[i] = (self.other_cons[ph], 0)

        return phones


    def __convert_vowels(self, phones: list) -> list:
        """Transcribe vowel phones.

            Args:
                phones (list): tuples of phones marked: 0 -- not stressed, 1 -- stressed.
                    Example: [('#', 0), ('s', 0), ('lj', 0), ('и', 1), ('v', 0), ('а', 0), ('#', 0)]

            Returns:
                list: consonants transcribed. Ex: ['#', 's', 'lj', 'i1', 'v', 'a0', '#']

        """
        new_phones = []
        prev = ""
        for (ph, stress) in phones:
            if prev in self.startsyl:
                if ph in set("яюеё"):
                    new_phones.append("j")
            if ph in self.vowels:
                new_phones.append(self.vowels[ph] + str(stress))
            else:
                new_phones.append(ph)
            prev = ph

        return new_phones


    def convert(self, stressword: str) -> str:
        """"""
        phones = ("#" + stressword + "#")

        # Assign stress marks
        stress_phones = []
        acc_before = False
        offset = -1 if self.acc_before else 1

        for i,ph in enumerate(phones[:-1]):
            if ph == '+': continue
            if phones[i+offset] == '+':
                stress_phones.append((ph,1))
            else:
                stress_phones.append((ph,0))
        else:
            stress_phones.append((phones[-1],0))

        phones = self.__convert_vowels(self.__pallatize(stress_phones))
        phones = [x for x in phones if x not in self.others]  # Filter
        return " ".join(phones)