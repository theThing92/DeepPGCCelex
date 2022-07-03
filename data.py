import pandas as pd
import torch
from logger import logger

class CELEXCorpus:

    def __init__(self,
                 path_to_excel="D:/Users/mvogel/Documents/SoSe 2022/"
                               "Effects of statistical learning on written language production/Projekt/data/"
                               "celex_berg_habil_belke.xlsx"):
        self.path_to_excel = path_to_excel
        self.df = self._load_data()
        logger.info(f"CELEXCorpus has been loaded from {self.path_to_excel}.")

    def _load_data(self):
        try:
            df = pd.read_excel(self.path_to_excel,
                               sheet_name="monomorph_de",
                               dtype={"id": "int32",
                                      "head": "string",
                                      "länge": "int32",
                                      "freq": "int32",
                                      "Status": "string",
                                      "<CV> mit Y=V": "string",
                                      "<CV>": "string",
                                      "phon_1": "string",
                                      "phon_2": "string",
                                      "phon_3": "string",
                                      "phon_4": "string",
                                      "CV1": "string",
                                      "CV2": "string",
                                      "gr_silbentrennung": "string",
                                      "<CV> ohne y": "string",
                                      "neue RS": "string"},
                               keep_default_na=False,
                               na_values=[""]
                               )
            df.set_index(["id"], inplace=True)
            df.drop_duplicates(inplace=True)
            # replace "$UMLAUT encoding with $UMLAUT
            df["gr_silbentrennung"] = df["gr_silbentrennung"]\
                .str.replace('"A', "Ä").str.replace('"O', "Ö").str.replace('"U', "Ü")\
                .str.replace('"a', "ä").str.replace('"o', "ö").str.replace('"u', "ü")
            # only select monomorphematic lexemes
            df = df[df["Status"] == "M"]

            return df

        except IOError as e:
            logger.error(e)
            raise e

    def save_data(self, path_to_save):
        try:
            self.df.to_csv(path_to_save, sep=",")
            logger.info(f"CELEX corpus has been saved to {path_to_save}.")

        except IOError as e:
            logger.error(e)
            raise e


class ChildLexCorpus:
    def __init__(self,
                 path_to_excel="D:/Users/mvogel/Documents/SoSe 2022/"
                               "Effects of statistical learning on written language production/Projekt/data/"
                               "childLex_0.17.01c_2018-12-24_schr.xlsx"):
        self.path_to_excel = path_to_excel
        self.df = self._load_data()
        logger.info(f"ChildLexCorpus has been loaded from {self.path_to_excel}.")

    def _load_data(self):
        try:
            df = pd.read_excel(self.path_to_excel,
                               sheet_name="All",
                               dtype={"type": "string",
                                      "pos": "string",
                                      "lemma": "string",
                                      "atype.abs": "int32",
                                      "atype.norm": "float32",
                                      "type.abs": "int32",
                                      "type.norm": "float32",
                                      "lemma.abs": "int32",
                                      "lemma.norm": "float32",
                                      "cd": "int32",
                                      "n.letters": "int32",
                                      "n.syl": "int32",
                                      "nei.n": "int32",
                                      "nei.old20": "float32",
                                      "bigram.sum": "int32",
                                      "bigram.min": "int32"},
                               keep_default_na = False,
                               na_values = [""]
                               )
            df.drop_duplicates(inplace=True)

            return df

        except IOError as e:
            logger.error(e)
            raise e

    def save_data(self, path_to_save):
        try:
            self.df.to_csv(path_to_save, sep=",")
            logger.info(f"ChildLexCorpus has been save to {path_to_save}.")

        except IOError as e:
            logger.error(e)
            raise e


def get_lemma_intersection_celex_childlex(celex_corpus, childlex_corpus):
    try:
        celex = celex_corpus.df
        childlex = childlex_corpus.df
        return celex[celex["head"].isin(childlex["lemma"].to_list())]

    except Exception as e:
        logger.error(e)
        raise e


def get_syllable_statistics(celex_corpus):
    celex_copy = celex_corpus.copy()

    vowels_graphemes = ["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"]
    umlaute_graphemes = ["ä", "ö", "ü", "Ä", "Ö", "Ü"]
    diphthongs_vowels_graphemes = ["au", "eu", "ei", "ie", "Au", "Eu", "Ei", "Ie"]
    diphthongs_umlaute_graphemes = ["äu", "Äu"]

    index_a = vowels_graphemes.index("a")
    letter_a = vowels_graphemes[index_a]
    index_A = vowels_graphemes.index("A")
    letter_A = vowels_graphemes[index_A]

    index_e = vowels_graphemes.index("e")
    letter_e = vowels_graphemes[index_e]
    index_E = vowels_graphemes.index("E")
    letter_E = vowels_graphemes[index_E]

    index_i = vowels_graphemes.index("i")
    letter_i = vowels_graphemes[index_i]
    index_I = vowels_graphemes.index("I")
    letter_I = vowels_graphemes[index_I]

    index_o = vowels_graphemes.index("o")
    letter_o = vowels_graphemes[index_o]
    index_O = vowels_graphemes.index("O")
    letter_O = vowels_graphemes[index_O]

    index_u = vowels_graphemes.index("u")
    letter_u = vowels_graphemes[index_u]
    index_U = vowels_graphemes.index("U")
    letter_U = vowels_graphemes[index_U]

    index_ae = umlaute_graphemes.index("ä")
    letter_ae = umlaute_graphemes[index_ae]
    index_AE = umlaute_graphemes.index("Ä")
    letter_AE = umlaute_graphemes[index_AE]

    index_oe = umlaute_graphemes.index("ö")
    letter_oe = umlaute_graphemes[index_oe]
    index_OE = umlaute_graphemes.index("Ö")
    letter_OE = umlaute_graphemes[index_OE]

    index_ue = umlaute_graphemes.index("ü")
    letter_ue = umlaute_graphemes[index_ue]
    index_UE = umlaute_graphemes.index("Ü")
    letter_UE = umlaute_graphemes[index_UE]

    index_au = diphthongs_vowels_graphemes.index("au")
    letter_au = diphthongs_vowels_graphemes[index_au]
    index_Au = diphthongs_vowels_graphemes.index("Au")
    letter_Au = diphthongs_vowels_graphemes[index_Au]

    index_eu = diphthongs_vowels_graphemes.index("eu")
    letter_eu = diphthongs_vowels_graphemes[index_eu]
    index_Eu = diphthongs_vowels_graphemes.index("Eu")
    letter_Eu = diphthongs_vowels_graphemes[index_Eu]

    index_ei = diphthongs_vowels_graphemes.index("ei")
    letter_ei = diphthongs_vowels_graphemes[index_ei]
    index_Ei = diphthongs_vowels_graphemes.index("Ei")
    letter_Ei = diphthongs_vowels_graphemes[index_Ei]

    index_ie = diphthongs_vowels_graphemes.index("ie")
    letter_ie = diphthongs_vowels_graphemes[index_ie]
    index_Ie = diphthongs_vowels_graphemes.index("Ie")
    letter_Ie = diphthongs_vowels_graphemes[index_Ie]

    index_aeu = diphthongs_umlaute_graphemes.index("äu")
    letter_aeu = diphthongs_umlaute_graphemes[index_aeu]
    index_AEu = diphthongs_umlaute_graphemes.index("Äu")
    letter_AEu = diphthongs_umlaute_graphemes[index_AEu]

    marker_stressed_syllable = "'"
    col_name_graphemes = "gr_silbentrennung"
    col_name_phonemes = "phon_1"

    freqs_stressed_unstressed = {
        "syllable_stressed_grapheme_vowels_total": 0,
        "syllable_stressed_grapheme_vowels_a": 0,
        "syllable_stressed_grapheme_vowels_e": 0,
        "syllable_stressed_grapheme_vowels_i": 0,
        "syllable_stressed_grapheme_vowels_o": 0,
        "syllable_stressed_grapheme_vowels_u": 0,
        "syllable_stressed_grapheme_umlaute_total": 0,
        "syllable_stressed_grapheme_umlaute_ä": 0,
        "syllable_stressed_grapheme_umlaute_ö": 0,
        "syllable_stressed_grapheme_umlaute_ü": 0,
        "syllable_stressed_grapheme_diphthongs_vowels_total": 0,
        "syllable_stressed_grapheme_diphthongs_vowels_au": 0,
        "syllable_stressed_grapheme_diphthongs_vowels_eu": 0,
        "syllable_stressed_grapheme_diphthongs_vowels_ei": 0,
        "syllable_stressed_grapheme_diphthongs_vowels_ie": 0,
        "syllable_stressed_grapheme_diphthongs_umlaute_äu": 0,
        "syllable_unstressed_grapheme_vowels_total": 0,
        "syllable_unstressed_grapheme_vowels_a": 0,
        "syllable_unstressed_grapheme_vowels_e": 0,
        "syllable_unstressed_grapheme_vowels_i": 0,
        "syllable_unstressed_grapheme_vowels_o": 0,
        "syllable_unstressed_grapheme_vowels_u": 0,
        "syllable_unstressed_grapheme_umlaute_total": 0,
        "syllable_unstressed_grapheme_umlaute_ä": 0,
        "syllable_unstressed_grapheme_umlaute_ö": 0,
        "syllable_unstressed_grapheme_umlaute_ü": 0,
        "syllable_unstressed_grapheme_diphthongs_vowels_total": 0,
        "syllable_unstressed_grapheme_diphthongs_vowels_au": 0,
        "syllable_unstressed_grapheme_diphthongs_vowels_eu": 0,
        "syllable_unstressed_grapheme_diphthongs_vowels_ei": 0,
        "syllable_unstressed_grapheme_diphthongs_vowels_ie": 0,
        "syllable_unstressed_grapheme_diphthongs_umlaute_äu": 0
    }

    try:
        # note: replace = by - for easier syllable segmentation
        celex_copy[col_name_graphemes] = celex_copy[col_name_graphemes].replace("=", "-", regex=True)
        celex_copy[col_name_graphemes] = celex_copy[col_name_graphemes].str.split("-")

        celex_copy[col_name_phonemes] = celex_copy[col_name_phonemes].str.split("-")

        graphemes = celex_copy[col_name_graphemes].to_list()
        phonemes = celex_copy[col_name_phonemes].to_list()

        graphemes_pruned = []
        phonemes_pruned = []

        # exclude lemma with no 1 to 1 syllable mapping for gr_silbentrennung and phon_1 in CELEX
        for i, graph in enumerate(graphemes):
            phon = phonemes[i]
            if len(graph) == len(phon):
                graphemes_pruned.append(graph)
                phonemes_pruned.append(phon)
            else:
                logger.info(f"Syllable mismatch between {phon} and {graph} in columns {col_name_phonemes} and {col_name_graphemes}"
                      f" with {len(phon)} syllables and {len(graph)} syllables."
                      f" The datum will be excluded from further analysis.")

        assert len(graphemes_pruned) == len(phonemes_pruned), "List indices between phoneme and grapheme lists are not equal."

        # iterate through syllables and count vowel, umlaut and diphthong freqs for stressed and unstressed syllables
        # note: for counting no distinction between letters in word-initial or non-word-initial syllables is made
        for i, phon in enumerate(phonemes_pruned):
            graph = graphemes_pruned[i]

            for j, syl in enumerate(phon):
                # stressed syllables
                if syl.startswith(marker_stressed_syllable):
                    for vowel in vowels_graphemes:
                        if vowel in graph[j]:
                            freqs_stressed_unstressed["syllable_stressed_grapheme_vowels_total"] += 1


                            if vowel in [letter_a,  letter_A] and \
                                    not (letter_au in graph[j] or letter_Au in graph[j]):
                                freqs_stressed_unstressed[
                                    f"syllable_stressed_grapheme_vowels_{letter_a}"] += 1

                            elif vowel in [letter_e, letter_E] and \
                                    not (letter_eu in graph[j] or letter_Eu in graph[j] or
                                         letter_ei in graph[j] or letter_Ei in graph[j] or
                                         letter_ie in graph[j] or letter_Ie in graph[j]):
                                freqs_stressed_unstressed[
                                    f"syllable_stressed_grapheme_vowels_{letter_e}"] += 1

                            elif vowel in [letter_i, letter_I] and \
                                    not (letter_ie in graph[j] or letter_Ie in graph[j] or
                                         letter_ei in graph[j] or letter_Ei in graph[j]):
                                freqs_stressed_unstressed[
                                    f"syllable_stressed_grapheme_vowels_{letter_i}"] += 1

                            elif vowel in [letter_o, letter_O]:
                                freqs_stressed_unstressed[
                                    f"syllable_stressed_grapheme_vowels_{letter_o}"] += 1

                            elif vowel in [letter_u, letter_U] and \
                                    not (letter_au in graph[j] or letter_Au in graph[j] or
                                         letter_eu in graph[j] or letter_Eu in graph[j] or
                                         letter_aeu in graph[j] or letter_AEu in graph[j]):
                                freqs_stressed_unstressed[
                                    f"syllable_stressed_grapheme_vowels_{letter_u}"] += 1

                    for umlaut in umlaute_graphemes:
                        if umlaut in graph[j]:
                            freqs_stressed_unstressed["syllable_stressed_grapheme_umlaute_total"] += 1

                            if umlaut in [letter_ae, letter_AE] and \
                                    not (letter_aeu in graph[j] or letter_AEu in graph[j]):
                                freqs_stressed_unstressed[
                                    f"syllable_stressed_grapheme_umlaute_{letter_ae}"] += 1

                            elif umlaut in [letter_oe, letter_OE]:
                                freqs_stressed_unstressed[
                                    f"syllable_stressed_grapheme_umlaute_{letter_oe}"] += 1

                            elif umlaut in [letter_ue, letter_UE]:
                                freqs_stressed_unstressed[
                                    f"syllable_stressed_grapheme_umlaute_{letter_ue}"] += 1

                    for diphthong_vowel in diphthongs_vowels_graphemes:
                        if diphthong_vowel in graph[j]:
                            freqs_stressed_unstressed["syllable_stressed_grapheme_diphthongs_vowels_total"] += 1

                            if diphthong_vowel in [letter_au, letter_Au]:
                                freqs_stressed_unstressed[
                                    f"syllable_stressed_grapheme_diphthongs_vowels_{letter_au}"] += 1

                            elif diphthong_vowel in [letter_eu, letter_Eu]:
                                freqs_stressed_unstressed[
                                    f"syllable_stressed_grapheme_diphthongs_vowels_{letter_eu}"] += 1

                            elif diphthong_vowel in [letter_ei, letter_Ei]:
                                freqs_stressed_unstressed[
                                    f"syllable_stressed_grapheme_diphthongs_vowels_{letter_ei}"] += 1

                            elif diphthong_vowel in [letter_ie, letter_Ie]:
                                freqs_stressed_unstressed[
                                    f"syllable_stressed_grapheme_diphthongs_vowels_{letter_ie}"] += 1

                    for diphthong_umlaut in diphthongs_umlaute_graphemes:
                        if diphthong_umlaut in graph[j]:
                            if diphthong_umlaut in [letter_aeu, letter_AEu]:
                                freqs_stressed_unstressed[
                                    f"syllable_stressed_grapheme_diphthongs_umlaute_{letter_aeu}"] += 1


                # unstressed syllables
                else:
                    for vowel in vowels_graphemes:
                        if vowel in graph[j]:
                            freqs_stressed_unstressed["syllable_unstressed_grapheme_vowels_total"] += 1


                            if vowel in [letter_a,  letter_A] and \
                                    not (letter_au in graph[j] or letter_Au in graph[j]):
                                freqs_stressed_unstressed[
                                    f"syllable_unstressed_grapheme_vowels_{letter_a}"] += 1

                            elif vowel in [letter_e, letter_E] and \
                                    not (letter_eu in graph[j] or letter_Eu in graph[j] or
                                         letter_ei in graph[j] or letter_Ei in graph[j] or
                                         letter_ie in graph[j] or letter_Ie in graph[j]):
                                freqs_stressed_unstressed[
                                    f"syllable_unstressed_grapheme_vowels_{letter_e}"] += 1

                            elif vowel in [letter_i, letter_I] and \
                                    not (letter_ie in graph[j] or letter_Ie in graph[j] or
                                         letter_ei in graph[j] or letter_Ei in graph[j]):
                                freqs_stressed_unstressed[
                                    f"syllable_unstressed_grapheme_vowels_{letter_i}"] += 1

                            elif vowel in [letter_o, letter_O]:
                                freqs_stressed_unstressed[
                                    f"syllable_unstressed_grapheme_vowels_{letter_o}"] += 1

                            elif vowel in [letter_u, letter_U] and \
                                    not (letter_au in graph[j] or letter_Au in graph[j] or
                                         letter_eu in graph[j] or letter_Eu in graph[j] or
                                         letter_aeu in graph[j] or letter_AEu in graph[j]):
                                freqs_stressed_unstressed[
                                    f"syllable_unstressed_grapheme_vowels_{letter_u}"] += 1

                    for umlaut in umlaute_graphemes:
                        if umlaut in graph[j]:
                            freqs_stressed_unstressed["syllable_unstressed_grapheme_umlaute_total"] += 1

                            if umlaut in [letter_ae, letter_AE] and \
                                    not (letter_aeu in graph[j] or letter_AEu in graph[j]):
                                freqs_stressed_unstressed[
                                    f"syllable_unstressed_grapheme_umlaute_{letter_ae}"] += 1

                            elif umlaut in [letter_oe, letter_OE]:
                                freqs_stressed_unstressed[
                                    f"syllable_unstressed_grapheme_umlaute_{letter_oe}"] += 1

                            elif umlaut in [letter_ue, letter_UE]:
                                freqs_stressed_unstressed[
                                    f"syllable_unstressed_grapheme_umlaute_{letter_ue}"] += 1

                    for diphthong_vowel in diphthongs_vowels_graphemes:
                        if diphthong_vowel in graph[j]:
                            freqs_stressed_unstressed["syllable_unstressed_grapheme_diphthongs_vowels_total"] += 1

                            if diphthong_vowel in [letter_au, letter_Au]:
                                freqs_stressed_unstressed[
                                    f"syllable_unstressed_grapheme_diphthongs_vowels_{letter_au}"] += 1

                            elif diphthong_vowel in [letter_eu, letter_Eu]:
                                freqs_stressed_unstressed[
                                    f"syllable_unstressed_grapheme_diphthongs_vowels_{letter_eu}"] += 1

                            elif diphthong_vowel in [letter_ei, letter_Ei]:
                                freqs_stressed_unstressed[
                                    f"syllable_unstressed_grapheme_diphthongs_vowels_{letter_ei}"] += 1

                            elif diphthong_vowel in [letter_ie, letter_Ie]:
                                freqs_stressed_unstressed[
                                    f"syllable_unstressed_grapheme_diphthongs_vowels_{letter_ie}"] += 1

                    for diphthong_umlaut in diphthongs_umlaute_graphemes:
                        if diphthong_umlaut in graph[j]:
                            if diphthong_umlaut in [letter_aeu, letter_AEu]:
                                freqs_stressed_unstressed[
                                    f"syllable_unstressed_grapheme_diphthongs_umlaute_{letter_aeu}"] += 1

        return freqs_stressed_unstressed



    except Exception as e:
        raise e


class PGCDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]

        return x, y



if __name__ == "__main__":
    celex = CELEXCorpus()
    childlex = ChildLexCorpus()
    intersection_lemma_celex_childlex = get_lemma_intersection_celex_childlex(celex, childlex)
    freqs = get_syllable_statistics(intersection_lemma_celex_childlex)
    intersection_lemma_celex_childlex.to_csv("data/pgc/intersection_celex_childlex_monomorp.csv", encoding="utf-8")

    # plot frequencies (not available in Pycharm due to bug)
    #import seaborn as sns
    #keys = list(freqs.keys())
    #vals = list(freqs.values())
    #barplot = sns.barplot(x=keys, y=vals)
    #fig = barplot.get_figure()

    #barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)
    #fig.savefig("barplot_freqs.png", bbox_inches ="tight")

