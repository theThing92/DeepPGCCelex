'''Module for rule-based phoneme to grapheme conversion (simple 1:1 alignment)'''
from dpgc import preprocessing
from dpgc import logger

# mappings according to consonant & vowel mappings in Eisenberg (2020) https://doi.org/10.1007/978-3-476-05096-0
# phonemes are encoded in DISC format (cf. CELEX German User Guide https://catalog.ldc.upenn.edu/docs/LDC96L14/gug_a4.pdf

CONVERSION_STRATEGIES = ["ignore_unknown"]

PGC = {
    # consonants
    "p": "p",
    "t": "t",
    "k": "k",
    "b": "b",
    "d": "d",
    "x": "ch",
    "v": "w",
    "j": "j",
    "h": "h",
    "m": "m",
    "g": "g",
    "f": "f",
    # note: Gla[s] not captured by context-free rules
    "z": "s",
    "s": "ß",
    "S": "sch",
    "n": "n",
    "N": "ng",
    "l": "l",
    "r": "r",

    # complex consonants
    "kv": "qu",
    "=": "z",
    "ks": "x",

    # vocals (stressed)
    "i": "ie",
    "y": "ü",
    "e": "e",
    "|": "ö",
    ")": "ä",
    "ɑ": "a",
    "o": "o",
    "u": "u",

    # vocals (unstressed)
    "I": "i",
    "Y": "ü",
    "E": "e",
    "/": "ö",
    "&": "a",
    "O": "o",
    "U": "u",

    # vocals (reduced)
    "@": "e",

    # diphthongs
    "W": "ei",
    "B": "au",
    "X": "eu"

}


def convert_phonemes2graphemes_pgc(phon_seq: str,
                                   strategy: str=CONVERSION_STRATEGIES[0],
                                   pgc_dict: dict = PGC,
                                   **kwargs):
    phon_seq_list_out = preprocessing.extract_tokens_list(phon_seq, **kwargs)

    if strategy:

        if strategy == CONVERSION_STRATEGIES[0]:
            graph_seq = ""

            for phon in phon_seq_list_out:
                try:
                    graph_seq += pgc_dict[phon]

                except KeyError:
                    pass

            return graph_seq

    elif not strategy:
        raise ValueError("No valid conversion strategy has been defined.")



if __name__ == "__main__":
    phonemes = "'&p-ve-z@nt"
    graphemes = convert_phonemes2graphemes_pgc(phonemes, complex_tokens = preprocessing.COMPLEX_PHONEMES,
                                               non_token_chars = preprocessing.NON_PHONEME_CHARS)
    logger.info(graphemes)


