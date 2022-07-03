import pandas as pd
import torch
import data
import numpy as np
from sklearn.model_selection import train_test_split
from logger import logger

COMPLEX_GRAPHEMES = []
NON_GRAPHEME_CHARS = ["-", "="]
COMPLEX_PHONEMES = ["kv", "ks"]
NON_PHONEME_CHARS = ["'", "-"]

def extract_tokens_list(tokens_seq: str, complex_tokens: list, non_token_chars: list, lower=True):
    token_seq_list_out = []

    if lower:
        tokens_seq = tokens_seq.lower()

    if complex_tokens:
        for ct in complex_tokens:
            tokens_seq = tokens_seq.replace(ct, "," + ct + ",")

        token_seq_list = tokens_seq.split(",")

        for i, token in enumerate(token_seq_list):
            if token_seq_list[i] not in complex_tokens:
                token_seq_list[i] = " ".join(token_seq_list[i]).split()
            else:
                token_seq_list[i] = [token_seq_list[i]]


        for token_list in token_seq_list:
            for token in token_list:
                token_seq_list_out.append(token)
    else:
        token_seq_list_out = " ".join(tokens_seq).split()

    if non_token_chars:
        token_seq_list_out = [char for char in token_seq_list_out if char not in non_token_chars]

    logger.info(f"Tokens for {tokens_seq} have been extracted.")

    return token_seq_list_out


def extract_phonemes(df: pd.DataFrame, **kwargs):
    try:
        phonemes = df["phon_1"].apply(extract_tokens_list, **kwargs).to_list()
        logger.info("Phonemes have been extracted.")
        return phonemes

    except Exception as e:
        raise e

def extract_graphemes(df: pd.DataFrame, **kwargs):
    try:
        graphemes = df["gr_silbentrennung"].apply(extract_tokens_list, **kwargs).to_list()
        logger.info("Graphemes have been extracted.")
        return graphemes

    except Exception as e:
        raise e

def get_mappings(X: list, pad_token="<PAD>"):
    mappings = {"<PAD>": 0}
    counter = 0

    for token_list in X:
        for token in token_list:
            if token not in mappings:
                counter += 1
                mappings[token] = counter
    logger.info("Matrix of token sequences have been mapped and padded.")

    return mappings


def map_tokens2index(X, mappings):
    tokens2index = [[mappings[token] for token in token_list] for token_list in X]
    logger.info("Tokens have been mapped to indices.")

    return tokens2index


def map_index2tokens(X, mappings):
    inverted_mappings = {v: k for k,v in mappings.items()}
    index2tokens = [[inverted_mappings[token] for token in token_list] for token_list in X]
    logger.info("Indices have been mapped to tokens.")

    return index2tokens


def pad_sequence(X, Y, mappings_X, mappings_Y, pad_token_X="<PAD>", pad_token_Y="<PAD>"):
    X_lengths = [len(phon_seq) for phon_seq in X]
    Y_lengths = [len(graph_seq) for graph_seq in Y]
    X_pad_token = mappings_X[pad_token_X]
    Y_pad_token = mappings_Y[pad_token_Y]


    longest_seq = max(map(max, [X_lengths, Y_lengths]))
    X_batch_size = len(X)
    Y_batch_size = len(Y)
    X_padded = np.ones((X_batch_size, longest_seq)) * X_pad_token
    Y_padded = np.ones((Y_batch_size, longest_seq)) * Y_pad_token

    for i, x_len in enumerate(X_lengths):
        sequence = X[i]
        X_padded[i, 0:x_len] = sequence[:x_len]

    for i, y_len in enumerate(Y_lengths):
        sequence = Y[i]
        Y_padded[i, 0:y_len] = sequence[:y_len]

    X_padded = torch.from_numpy(X_padded).to(torch.long)
    Y_padded = torch.from_numpy(Y_padded).to(torch.long)

    logger.info(f"Sequences for X and Y have been padded to {longest_seq} chars.")

    return X_padded, Y_padded


def one_hot_encode(indices, dict_size):
    ''' Define one hot encode matrix for our sequences'''
    # Creating a multi-dimensional array with the desired output shape
    # Encode every integer with its one hot representation
    features = np.eye(dict_size, dtype=np.float32)[np.array(indices).flatten()]

    # Finally reshape it to get back to the original array
    features = features.reshape((*indices.shape, dict_size))

    # TODO: pytorch tensor to one hot encoding
    # F.one_hot(torch.tensor(tokens2index_phonemes_padded).to(torch.int64))

    return features



def get_train_dev_test_data(X, Y, train_ratio = 0.8, dev_ratio=0.1, test_ratio=0.1, random_state=42):
    assert train_ratio + dev_ratio + test_ratio == 1.0, "Check if train_ratio, dev_ratio and test_ratio sum up to 1.0."


    try:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=test_ratio, random_state=random_state)
        split_ratio = dev_ratio / train_ratio
        X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, shuffle=True, test_size=split_ratio, random_state=random_state)
        return X_train, X_dev, X_test, Y_train, Y_dev, Y_test

    except Exception as e:
        logger.error(e)
        raise e




if __name__ == "__main__":
    celex = data.CELEXCorpus()
    childlex = data.ChildLexCorpus()
    intersection_lemma_celex_childlex = data.get_lemma_intersection_celex_childlex(celex, childlex)
    freqs = data.get_syllable_statistics(intersection_lemma_celex_childlex)
    intersection_lemma_celex_childlex.to_csv("data/pgc/intersection_celex_childlex_monomorp.csv", encoding="utf-8")

    phonemes = extract_phonemes(intersection_lemma_celex_childlex, complex_tokens=COMPLEX_PHONEMES, non_token_chars=NON_PHONEME_CHARS)
    graphemes = extract_graphemes(intersection_lemma_celex_childlex, complex_tokens=COMPLEX_GRAPHEMES, non_token_chars=NON_GRAPHEME_CHARS)

    vocab_phonemes = get_mappings(phonemes)
    vocab_graphemes = get_mappings(graphemes)

    tokens2index_phonemes = map_tokens2index(phonemes, vocab_phonemes)
    tokens2index_graphemes = map_tokens2index(graphemes, vocab_graphemes)

    tokens2index_phonemes_padded, tokens2index_graphemes_padded = pad_sequence(tokens2index_phonemes,
                                                                               tokens2index_graphemes,
                                                                               vocab_phonemes,
                                                                               vocab_graphemes)

    index2tokens_phonemes =  map_index2tokens(tokens2index_phonemes, vocab_phonemes)
    index2tokens_graphemes = map_index2tokens(tokens2index_graphemes, vocab_graphemes)

    X_train, X_dev, X_test, Y_train, Y_dev, Y_test = get_train_dev_test_data(tokens2index_phonemes_padded, tokens2index_graphemes_padded)


    dataset_train = data.PGCDataset(X_train, Y_train)
    dataset_dev = data.PGCDataset(X_dev, Y_dev)
    dataset_test = data.PGCDataset(X_test, Y_test)

    # transform phonemes / graphemes in to corresponding one-hot-vectors
    # add padding function for equal length input (padding char = *)
