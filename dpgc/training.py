import os
import torch
from dpgc import preprocessing, data, configuration, model
from dpgc import logger
from dpgc import device



def train_model(dataset_train, dataset_dev, dataset_test, model, configuration):
    torch.backends.cudnn.benchmark = True

    # Parameters
    data_generator_params = {"batch_size": configuration.configuration["batch_size"],
                             "num_workers": configuration.configuration["num_workers"]}

    # Generators
    generator_train = torch.utils.data.DataLoader(dataset_train, **data_generator_params, shuffle=True)
    generator_dev = torch.utils.data.DataLoader(dataset_dev, **data_generator_params, shuffle=False)
    generator_test = torch.utils.data.DataLoader(dataset_test, **data_generator_params, shuffle=False)


    #loss_function = torch.nn.CTCLoss(blank=model.phonemes2index[model.pad_token], reduction="mean")
    loss_function = model.loss
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=configuration.configuration["learning_rate"],
                                weight_decay=configuration.configuration["weight_decay"],
                                momentum=configuration.configuration["momentum"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="min",
                                                           factor=configuration.configuration["factor"],
                                                           patience=configuration.configuration["patience"],
                                                           threshold=configuration.configuration["threshold"],
                                                           verbose=True)

    logger.info("Starting training with following model architecture...")
    logger.info(f"{model}")

    temp_val_loss = 99999999999

    mean_train_losses = []
    mean_dev_losses = []
    mean_test_losses = []

    # Loop over epochs
    for epoch in range(1, configuration.configuration["max_epochs"]+1):
        logger.info(f"Epoch {epoch}...")

        mean_train_loss = 0
        logger.info("Training model on train data.")
        # Training
        for i, (X, Y) in enumerate(generator_train):
            logger.info(f"Processing train batch {i+1} / {len(generator_train)}...")
            # Transfer to GPU
            X, Y = X.to(device), Y.to(device)
            X_lengths = [X.shape[-1] for arr in range(X.shape[0])]
            # Model computations
            model.train(True)
            optimizer.zero_grad()
            Y_pred = model(X, X_lengths)
            train_loss_batch = loss_function(Y_pred, Y)
            train_loss_batch.backward()
            optimizer.step()

            mean_train_loss += train_loss_batch.item() / len(generator_train)
        logger.info(f"Mean train loss is: {mean_train_loss}.")
        mean_train_losses.append(mean_train_loss)

        mean_dev_loss = 0
        model.eval()
        logger.info("Testing model on dev data.")
        # Validation
        with torch.set_grad_enabled(False):
            for i, (X, Y) in enumerate(generator_dev):
                logger.info(f"Processing dev batch {i+1} / {len(generator_dev)}...")
                # Transfer to GPU
                X, Y = X.to(device), Y.to(device)
                X_lengths = [X.shape[-1] for arr in range(X.shape[0])]

                # Model computations
                Y_pred = model(X, X_lengths)
                dev_loss_batch = loss_function(Y_pred, Y)
                mean_dev_loss += dev_loss_batch.item() / len(generator_dev)

            if mean_dev_loss < temp_val_loss:
                temp_val_loss = mean_dev_loss

            scheduler.step(mean_dev_loss)
        logger.info(f"Mean dev loss is: {mean_dev_loss}.")
        mean_dev_losses.append(mean_dev_loss)
        if epoch % configuration.configuration["save_model_each_n_epochs"] == 0:
            try:
                model_name = f"model_epoch_{epoch}.pickle"
                if not os.path.exists(configuration.configuration["model_directory"]):
                    os.mkdir(configuration.configuration["model_directory"])
                    logger.info(f"Model directory {configuration.configuration['model_directory']} does not exist. "
                                f"Creating directory at given path.")
                model_filepath = os.path.join(os.path.abspath(configuration.configuration["model_directory"]), model_name)
                torch.save(model.state_dict(), model_filepath)
                config_filepath = os.path.join(os.path.abspath(configuration.configuration["model_directory"]), "config.yaml")
                configuration.export_configuration(config_filepath)
                logger.info(f"Model for epoch {epoch} has been saved at {model_filepath}.")

            except Exception as e:
                logger.error(e)
                raise e

        # test model
        mean_test_loss = 0
        logger.info(f"Testing model on held out dataset.")
        for i, (X, Y) in enumerate(generator_test):
            logger.info(f"Processing test batch {i+1} / {len(generator_test)}...")
            # Transfer to GPU
            X, Y = X.to(device), Y.to(device)
            X_lengths = [X.shape[-1] for arr in range(X.shape[0])]

            # Model computations
            Y_pred = model(X, X_lengths)
            test_loss = loss_function(Y_pred, Y)
            mean_test_loss += test_loss.item() / len(generator_test)
        logger.info(f"Mean test loss is: {mean_test_loss}.")
        mean_test_losses.append(mean_test_loss)

    return mean_train_losses, mean_dev_losses, mean_test_losses



if __name__ == "__main__":
    # Set a seed to reproduce experiments
    torch.manual_seed(42)

    bilistm_config = configuration.BiPGCLSTMConfiguration()

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

    # TODO: make configuration easier
    bilistm_model = model.BiPGCLSTM(phonemes2index=vocab_phonemes,
                                    graphemes2index=vocab_graphemes,
                                    pad_token=bilistm_config.configuration["pad_token"],
                                    num_layers_lstm=bilistm_config.configuration["num_layers_lstm"],
                                    num_lstm_units=bilistm_config.configuration["num_lstm_units"],
                                    embedding_dim=bilistm_config.configuration["embedding_dim"],
                                    batch_size=bilistm_config.configuration["batch_size"])

    bilistm_model = bilistm_model.to(device)


    mean_train_losses, mean_dev_losses, mean_test_losses =  train_model(dataset_train,
                                                                        dataset_dev,
                                                                        dataset_test,
                                                                        bilistm_model,
                                                                        bilistm_config)
    header = ["epoch", "train_loss", "dev_loss", "test_loss"]
    losses_filename = "losses.csv"
    losses_filepath = os.path.join(os.path.abspath(bilistm_config.configuration["model_directory"]), losses_filename)

    with open(losses_filepath, "w", encoding="utf-8") as f:
        print(",".join(header), file=f)

        for i, _ in enumerate(mean_train_losses):
            epoch = i+1
            line = [str(epoch), str(mean_train_losses[i]), str(mean_dev_losses[i]), str(mean_test_losses[i])]
            print(",".join(line), file=f)





