import json
import math
import os

import numpy as np


def read_data(path_data='/data/datasets/soccernet/SoccerNet-code/src', shuffle=False):
    old_data_train_path_list = np.load(os.path.join(path_data, 'listgame_Train_300.npy'))
    old_data_valid_path_list = np.load(os.path.join(path_data, 'listgame_Valid_100.npy'))
    old_data_test_path_list = np.load(os.path.join(path_data, 'listgame_Test_100.npy'))

    if not shuffle:
        return old_data_train_path_list, old_data_valid_path_list, old_data_test_path_list

    list_games = np.append(np.append(old_data_train_path_list, old_data_valid_path_list), old_data_test_path_list)

    # np.random.seed(50)
    np.random.shuffle(list_games)

    train_games = list_games[0:300]
    valid_games = list_games[300:400]
    test_games = list_games[400:500]

    return train_games, valid_games, test_games


def dataset_generator(list_games, path_data, feature_type, batch_size, data_aug=True, chunk_size=60, PCA=True):
    FEATURE_PER_SECOND = 2
    feature_per_chunk = FEATURE_PER_SECOND * chunk_size

    if PCA:
        n_features = 512
    elif "ResNET" in feature_type:
        n_features = 2048
    elif "C3D" in feature_type:
        n_features = 4096
    elif "I3D" in feature_type:
        n_features = 1024

    feature_dict = {}
    label_dict = {}
    n_sample = 0
    while True:
        for game in list_games:
            game_path = os.path.join(path_data, game)
            for feature_game_name in os.listdir(game_path):
                if feature_type in feature_game_name and (
                        (PCA and "PCA" in feature_game_name) or (not PCA and "PCA" not in feature_game_name)):
                    feature_game_path = os.path.join(game_path, feature_game_name)

                    n_sample += 1

                    current_features_rough = np.load(feature_game_path)

                    label_game_path = os.path.join(game_path, "Labels.json")
                    with open(label_game_path) as labelFile:
                        jsonLabel = json.loads(labelFile.read())

                    num_aug = 0
                    for event in jsonLabel["annotations"]:
                        time_half = int(event["gameTime"][0])
                        time_minute = int(event["gameTime"][-5:-3])
                        time_second = int(event["gameTime"][-2:])

                        if (("1_" in feature_game_name and time_half == 1) or (
                                "2_" in feature_game_name and time_half == 2)) and (data_aug is True):

                            t = time_minute * 60 + time_second
                            t_ini = t - int(chunk_size * 0.3 / 2.0)
                            f_ini = t_ini * FEATURE_PER_SECOND
                            t_end = t + int(chunk_size * 0.3 / 2.0)
                            f_end = t_end * FEATURE_PER_SECOND

                            for f in range(f_ini, f_end, 1):
                                if (f + chunk_size < len(current_features_rough)) and (f - chunk_size > 0):
                                    num_aug += 1

                    current_features = np.zeros(
                        (int(math.ceil(current_features_rough.shape[0] / feature_per_chunk) + num_aug),
                         feature_per_chunk, current_features_rough.shape[1]), dtype=np.float32)

                    for i in range(current_features_rough.shape[0]):
                        current_features[i // feature_per_chunk, i % feature_per_chunk, :] = current_features_rough[i]
                        next_feat_aug = i // feature_per_chunk + 1

                    feature_dict[feature_game_path] = current_features

                    label = np.zeros((current_features.shape[0], 4), dtype=np.uint8)
                    label[:, 0] = 1

                    for event in jsonLabel["annotations"]:
                        time_half = int(event["gameTime"][0])
                        time_minute = int(event["gameTime"][-5:-3])
                        time_second = int(event["gameTime"][-2:])

                        if "card" in event["label"]:
                            curr_label = 1
                        elif "subs" in event["label"]:
                            curr_label = 2
                        elif "soccer" in event["label"]:
                            curr_label = 3

                        if ("1_" in feature_game_name and time_half == 1) or (
                                "2_" in feature_game_name and time_half == 2):
                            time = time_minute * 60 + time_second
                            index = min(int(time / chunk_size), label.shape[0] - 1)
                            label[index, 0] = 0
                            label[index, curr_label] = 1

                            if data_aug is True:
                                t = time_minute * 60 + time_second
                                t_ini = t - int(chunk_size * 0.3 / 2.0)
                                f_ini = t_ini * FEATURE_PER_SECOND
                                t_end = t + int(chunk_size * 0.3 / 2.0)
                                f_end = t_end * FEATURE_PER_SECOND

                                for f in range(f_ini, f_end, 1):
                                    if (f + chunk_size < len(current_features_rough)) and (f - chunk_size > 0):
                                        extra_feat = current_features_rough[f - chunk_size:f + chunk_size, :]
                                        current_features[next_feat_aug] = extra_feat

                                        label[next_feat_aug, 0] = 0
                                        label[next_feat_aug, curr_label] = 1
                                        next_feat_aug += 1

                    label_dict[feature_game_path] = label

                    if n_sample % batch_size == 0:
                        samples = np.zeros((1, feature_per_chunk, n_features), dtype=np.float32)
                        targets = np.zeros((1, 4), dtype=np.uint8)
                        targets[0] = [1, 0, 0, 0]
                        for k, v in feature_dict.items():
                            samples = np.append(samples, v, axis=0)
                            targets = np.append(targets, label_dict[k], axis=0)

                        yield samples, targets

                        feature_dict = {}
                        label_dict = {}


def generators(path_data, feature_type, batch_size_train, batch_size_val_and_test, data_aug=False, chunk_size=60,
               PCA=True):
    train_games, validation_games, test_games = read_data(path_data)
    train_gen = dataset_generator(train_games, "/data/datasets/soccernet/SoccerNet-code/data", feature_type,
                                  batch_size_train,
                                  data_aug, chunk_size, PCA)
    validation_gen = dataset_generator(validation_games, "/data/datasets/soccernet/SoccerNet-code/data", feature_type,
                                       batch_size_val_and_test, False, chunk_size, PCA)
    test_gen = dataset_generator(test_games, "/data/datasets/soccernet/SoccerNet-code/data", feature_type,
                                 batch_size_val_and_test,
                                 False, chunk_size, PCA)
    return train_gen, validation_gen, test_gen
