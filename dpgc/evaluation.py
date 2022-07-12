import os
import torch
import pandas as pd
from dpgc import logger
from dpgc import model
from dpgc import device


def test_model(dataset_train, dataset_dev, dataset_test, model, configuration):
    torch.backends.cudnn.benchmark = True

    # Parameters
    data_generator_params = {"batch_size": configuration.configuration["batch_size"],
                             "num_workers": configuration.configuration["num_workers"]}

    # Generators
    generator_train = torch.utils.data.DataLoader(dataset_train, **data_generator_params, shuffle=False)
    generator_dev = torch.utils.data.DataLoader(dataset_dev, **data_generator_params, shuffle=False)
    generator_test = torch.utils.data.DataLoader(dataset_test, **data_generator_params, shuffle=False)


    logger.info("Testing the following model architecture...")
    logger.info(f"{model}")

    pred_train = []
    pred_dev = []
    pred_test = []

    with torch.no_grad():
        logger.info("Testing model on train data.")
        model.eval()
        for i, (X, Y) in enumerate(generator_train):
            logger.info(f"Processing train batch {i+1} / {len(generator_train)}...")
            # Transfer to GPU
            X, Y = X.to(device), Y.to(device)
            X_lengths = [X.shape[-1] for arr in range(X.shape[0])]
            # Model computations
            Y_pred = model(X, X_lengths)

            for y in Y_pred:
                y = y.argmax(dim=1).tolist()
                pred_train.append(y)

        logger.info("Testing model on dev data.")
        # Validation
        for i, (X, Y) in enumerate(generator_dev):
            logger.info(f"Processing dev batch {i+1} / {len(generator_dev)}...")
            # Transfer to GPU
            X, Y = X.to(device), Y.to(device)
            X_lengths = [X.shape[-1] for arr in range(X.shape[0])]

            # Model computations
            Y_pred = model(X, X_lengths)

            for y in Y_pred:
                y = y.argmax(dim=1).tolist()
                pred_dev.append(y)

        logger.info(f"Testing model on held out dataset.")
        for i, (X, Y) in enumerate(generator_test):
            logger.info(f"Processing test batch {i+1} / {len(generator_test)}...")
            # Transfer to GPU
            X, Y = X.to(device), Y.to(device)
            X_lengths = [X.shape[-1] for arr in range(X.shape[0])]

            # Model computations
            Y_pred = model(X, X_lengths)

            for y in Y_pred:
                y = y.argmax(dim=1).tolist()
                pred_test.append(y)

    return pred_train, pred_dev, pred_test


def levenshtein(s1, s2, normalized=True):
    if s1 == s2:
        levenshtein = 0
        return levenshtein
    elif len(s1) == 0:
        levenshtein = len(s2)
        return levenshtein
    elif len(s2) == 0:
        levenshtein = len(s1)
        return levenshtein
    v0 = [None] * (len(s2) + 1)
    v1 = [None] * (len(s2) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]
    levenshtein = v1[len(s2)]
    if normalized:
        levenshtein /= max(len(s1), len(s2))

    return levenshtein

def convert_index2string(indices, index2token_mappings):
    s = ""
    # TODO: as param
    pad_token = index2token_mappings[0]
    try:
        for index in indices:
            s += index2token_mappings[index]

    except IndexError:
        logger.info(f"Index {index} has no mapping in dictionary.")
    s = s.replace(pad_token, "")
    return s


def generate_evaluation(dataset_train, dataset_dev, dataset_test, configuration, phonemes2index, graphemes2index):
    index2graphemes_mappings = preprocessing.invert_mappings(graphemes2index)
    header = ["TARGET"]
    filename_evaluation_train = "evaluation_train.csv"
    filename_evaluation_dev = "evaluation_dev.csv"
    filename_evaluation_test = "evaluation_test.csv"
    model_directory = configuration.configuration["model_directory"]
    evaluation_filepath_train = os.path.join(os.path.abspath(model_directory), filename_evaluation_train)
    evaluation_filepath_dev = os.path.join(os.path.abspath(model_directory), filename_evaluation_dev)
    evaluation_filepath_test = os.path.join(os.path.abspath(model_directory), filename_evaluation_test)

    filename_models = sorted([f for f in os.listdir(model_directory) if f.endswith(".pickle")], key=lambda x: int(x.replace(".pickle", "").split("_")[-1]))
    filepath_models = [os.path.join(os.path.abspath(model_directory), f) for f in filename_models]

    target_indices_train = [y.tolist() for i, (x, y) in enumerate(dataset_train)]
    target_indices_dev = [y.tolist() for i, (x, y) in enumerate(dataset_dev)]
    target_indices_test = [y.tolist() for i, (x, y) in enumerate(dataset_test)]

    target_string_train = [convert_index2string(l, index2graphemes_mappings) for l in target_indices_train]
    target_string_dev = [convert_index2string(l, index2graphemes_mappings) for l in target_indices_dev]
    target_string_test = [convert_index2string(l, index2graphemes_mappings) for l in target_indices_test]

    pred_string_train = [[] for _ in range(len(filepath_models))]
    pred_string_dev = [[] for _ in range(len(filepath_models))]
    pred_string_test = [[] for _ in range(len(filepath_models))]

    pred_levenshtein_train = [[] for _ in range(len(filepath_models))]
    pred_levenshtein_dev = [[] for _ in range(len(filepath_models))]
    pred_levenshtein_test = [[] for _ in range(len(filepath_models))]

    header_model_id_pred = []
    header_model_id_levenshtein = []

    for i, model_filepath in enumerate(filepath_models):
        model_id = "_".join(os.path.basename(model_filepath).replace(".pickle", "").split("_")).upper()
        logger.info(f"Evaluating model with id {model_id}...")
        model_id_levenshtein = "LEVENSHTEIN_" + model_id
        model_id_prediction = "PREDICTION_" + model_id
        header_model_id_pred.append(model_id_prediction)
        header_model_id_levenshtein.append(model_id_levenshtein)

        m = model.BiPGCLSTM(phonemes2index=phonemes2index,
                                graphemes2index=graphemes2index,
                                pad_token=configuration.configuration["pad_token"],
                                num_layers_lstm=configuration.configuration["num_layers_lstm"],
                                num_lstm_units=configuration.configuration["num_lstm_units"],
                                embedding_dim=configuration.configuration["embedding_dim"],
                                batch_size=configuration.configuration["batch_size"])
        m.load_state_dict(torch.load(model_filepath))
        m = m.to(device)
        train_pred, dev_pred, test_pred = test_model(dataset_train, dataset_dev, dataset_test, m, configuration)

        train_string = [convert_index2string(l, index2graphemes_mappings) for l in train_pred]
        dev_string = [convert_index2string(l, index2graphemes_mappings) for l in dev_pred]
        test_string = [convert_index2string(l, index2graphemes_mappings) for l in test_pred]
        train_levenshtein = [levenshtein(s_pred, target_string_train[i]) for i, s_pred in enumerate(train_string)]
        dev_levenshtein = [levenshtein(s_pred, target_string_dev[i]) for i, s_pred in enumerate(dev_string)]
        test_levenshtein = [levenshtein(s_pred, target_string_test[i]) for i, s_pred in enumerate(test_string)]

        pred_string_train[i] = train_string
        pred_levenshtein_train[i] = train_levenshtein

        pred_string_dev[i] = dev_string
        pred_levenshtein_dev[i] = dev_levenshtein

        pred_string_test[i] = test_string
        pred_levenshtein_test[i] = test_levenshtein
        logger.info(f"Evaluation for model with id {model_id} is finished, deleting model.")
        del m

    header_out = []
    header_out += header + header_model_id_pred + header_model_id_levenshtein

    data_out_train = list(zip(target_string_train, *pred_string_train, *pred_levenshtein_train))
    data_out_dev = list(zip(target_string_dev, *pred_string_dev, *pred_levenshtein_dev))
    data_out_test = list(zip(target_string_test, *pred_string_test, *pred_levenshtein_test))

    df_train = pd.DataFrame(data_out_train, columns=header_out)
    df_dev = pd.DataFrame(data_out_dev, columns=header_out)
    df_test = pd.DataFrame(data_out_test, columns=header_out)

    return df_train, df_dev, df_test







if __name__ == "__main__":
    import dpgc.data as data
    import dpgc.configuration as configuration
    import dpgc.preprocessing as preprocessing

    # Set a seed to reproduce experiments
    torch.manual_seed(42)

    bilistm_config = configuration.BiPGCLSTMConfiguration("data/model/bipgclstm/config.yaml")

    celex = data.CELEXCorpus()
    childlex = data.ChildLexCorpus()
    intersection_lemma_celex_childlex = data.get_lemma_intersection_celex_childlex(celex, childlex)
    freqs = data.get_syllable_statistics(intersection_lemma_celex_childlex)
    intersection_lemma_celex_childlex.to_csv("data/pgc/intersection_celex_childlex_monomorp.csv", encoding="utf-8")

    phonemes = preprocessing.extract_phonemes(intersection_lemma_celex_childlex, complex_tokens=preprocessing.COMPLEX_PHONEMES, non_token_chars=preprocessing.NON_PHONEME_CHARS)
    graphemes = preprocessing.extract_graphemes(intersection_lemma_celex_childlex, complex_tokens=preprocessing.COMPLEX_GRAPHEMES, non_token_chars=preprocessing.NON_GRAPHEME_CHARS)

    vocab_phonemes = preprocessing.get_mappings(phonemes)
    vocab_graphemes = preprocessing.get_mappings(graphemes)

    vocab_phonemes_inverted = preprocessing.invert_mappings(vocab_phonemes)
    vocab_graphemes_inverted = preprocessing.invert_mappings(vocab_graphemes)

    tokens2index_phonemes = preprocessing.map_tokens2index(phonemes, vocab_phonemes)
    tokens2index_graphemes = preprocessing.map_tokens2index(graphemes, vocab_graphemes)

    tokens2index_phonemes_padded, tokens2index_graphemes_padded = preprocessing.pad_sequence(tokens2index_phonemes,
                                                                                             tokens2index_graphemes,
                                                                                             vocab_phonemes,
                                                                                             vocab_graphemes)

    index2tokens_phonemes =  preprocessing.map_index2tokens(tokens2index_phonemes, vocab_phonemes)
    index2tokens_graphemes = preprocessing.map_index2tokens(tokens2index_graphemes, vocab_graphemes)

    X_train, X_dev, X_test, Y_train, Y_dev, Y_test = preprocessing.get_train_dev_test_data(tokens2index_phonemes_padded,
                                                                                           tokens2index_graphemes_padded,
                                                                                           train_ratio=bilistm_config.configuration["train_ratio"],
                                                                                           dev_ratio=bilistm_config.configuration["dev_ratio"],
                                                                                           test_ratio=bilistm_config.configuration["test_ratio"],
                                                                                           random_state=bilistm_config.configuration["random_state"])
    dataset_train = data.PGCDataset(X_train, Y_train)
    dataset_dev = data.PGCDataset(X_dev, Y_dev)
    dataset_test = data.PGCDataset(X_test, Y_test)

    df_pred_train, df_pred_dev, df_pred_test = generate_evaluation(dataset_train, dataset_dev, dataset_test, bilistm_config, vocab_phonemes, vocab_graphemes)

    df_pred_train.sort_values(by="TARGET").to_csv("data/model/bipgclstm/pred_train.csv", encoding="utf-8")
    df_pred_dev.sort_values(by="TARGET").to_csv("data/model/bipgclstm/pred_dev.csv", encoding="utf-8")
    df_pred_test.sort_values(by="TARGET").to_csv("data/model/bipgclstm/pred_test.csv", encoding="utf-8")
