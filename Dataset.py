import json
import math
import os

import numpy as np


def read_shuffle_data(path_data):
    old_data_test_path_list = np.load(os.path.join(path_data, 'listgame_Test_100.npy'))
    old_data_train_path_list = np.load(os.path.join(path_data, 'listgame_Train_300.npy'))
    old_data_valid_path_list = np.load(os.path.join(path_data, 'listgame_Valid_100.npy'))

    list_games = np.append(np.append(old_data_test_path_list, old_data_train_path_list), old_data_valid_path_list)

    np.random.seed(50)

    np.random.shuffle(list_games)

    return list_games


def find_tensor_dim(list_games, path_data, feature_type, feature_per_chunk, PCA):
    max_chunk_feature_per_match = 0
    n_features = 0
    for game in list_games:
        game_path = os.path.join(path_data, game)
        for feature_game_name in os.listdir(game_path):
            if feature_type in feature_game_name and (
                    (PCA and "PCA" in feature_game_name) or (not PCA and "PCA" not in feature_game_name)):
                feature_game_path = os.path.join(game_path, feature_game_name)

                current_features_rough = np.load(feature_game_path)

                current_features = np.zeros((int(math.ceil(current_features_rough.shape[0] / feature_per_chunk)),
                                             feature_per_chunk, current_features_rough.shape[1]))

                max_chunk_feature_per_match = max(max_chunk_feature_per_match, current_features.shape[0])
                n_features = current_features_rough.shape[1]

    return max_chunk_feature_per_match, n_features


def dataset_generator(list_games, path_data, feature_type, batch_size, chunk_size=60, PCA=True):
    FEATURE_PER_SECOND = 2
    feature_per_chunk = FEATURE_PER_SECOND * chunk_size

    '''old_data_test_path_list = np.load(os.path.join(path_data, 'listgame_Test_100.npy'))
    old_data_train_path_list = np.load(os.path.join(path_data, 'listgame_Train_300.npy'))
    old_data_valid_path_list = np.load(os.path.join(path_data, 'listgame_Valid_100.npy'))

    list_games = np.append(np.append(old_data_test_path_list, old_data_train_path_list), old_data_valid_path_list)

    # list_games = np.load(path_list)
    '''

    max_chunk_feature_per_match, n_features = find_tensor_dim(list_games, path_data, feature_type, feature_per_chunk,
                                                              PCA)

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
                    # length = current_features_rough.shape[0] - current_features_rough.shape[0] % feature_per_chunk
                    # print(length)
                    # current_features = current_features_rough.reshape(int(length / feature_per_chunk)+1, feature_per_chunk,
                    # current_features_rough.shape[1])
                    current_features = np.zeros((int(math.ceil(current_features_rough.shape[0] / feature_per_chunk)),
                                                 feature_per_chunk, current_features_rough.shape[1]))

                    for i in range(current_features_rough.shape[0]):
                        current_features[i // feature_per_chunk, i % feature_per_chunk, :] = current_features_rough[i]

                    feature_dict[feature_game_path] = current_features

                    label = np.zeros((current_features.shape[0], 4), dtype=int)
                    label[:, 0] = 1
                    label_game_path = os.path.join(game_path, "Labels.json")
                    with open(label_game_path) as labelFile:
                        jsonLabel = json.loads(labelFile.read())

                    for event in jsonLabel["annotations"]:
                        time_half = int(event["gameTime"][0])
                        time_minute = int(event["gameTime"][-5:-3])
                        time_second = int(event["gameTime"][-2:])

                        if ("card" in event["label"]):
                            curr_label = 1
                        elif ("subs" in event["label"]):
                            curr_label = 2
                        elif ("soccer" in event["label"]):
                            curr_label = 3

                        if ("1_" in feature_game_name and time_half == 1) or (
                                "2_" in feature_game_name and time_half == 2):
                            time = time_minute * 60 + time_second
                            index = min(int(time / chunk_size), label.shape[0] - 1)
                            label[index, 0] = 0
                            label[index, curr_label] = 1

                        label_dict[feature_game_path] = label

                    if n_sample % batch_size == 0:
                        features_per_game = np.zeros((len(feature_dict), max_chunk_feature_per_match, feature_per_chunk,
                                                      n_features))
                        labels_per_game = np.zeros((len(label_dict), max_chunk_feature_per_match, 4), dtype=int)
                        i = 0
                        for k, v in feature_dict.items():
                            num_chunk_features = v.shape[0]
                            features_per_game[i, 0:num_chunk_features] = v
                            labels_per_game[i, 0:num_chunk_features] = label_dict[k]
                            i += 1
                        samples = np.reshape(features_per_game, (-1, feature_per_chunk, n_features))
                        targets = np.reshape(labels_per_game, (-1, 4))

                        yield samples, targets

                        feature_dict = {}
                        label_dict = {}


def generators(path_data, feature_type, batch_size, chunk_size=60, PCA=True):
    list_games = read_shuffle_data(path_data)
    train_games = list_games[0:300]
    validation_games = list_games[300:400]
    test_games = list_games[400:500]
    train_gen = dataset_generator(train_games, path_data, feature_type, batch_size, chunk_size, PCA)
    validation_gen = dataset_generator(validation_games, path_data, feature_type, batch_size, chunk_size, PCA)
    test_gen = dataset_generator(test_games, path_data, feature_type, batch_size, chunk_size, PCA)
    return train_gen, validation_gen, test_gen

    '''all_features_divided_per_game = np.zeros((len(feature_dict), max_chunk_feature_per_match, feature_per_chunk,
                                            ##  n_features))
    all_label_divided_per_game = np.zeros((len(label_dict), max_chunk_feature_per_match, 4), dtype=int)

    i = 0
    for k, v in feature_dict.items():
        num_chunk_features = v.shape[0]
        all_features_divided_per_game[i, 0:num_chunk_features] = v
        all_label_divided_per_game[i, 0:num_chunk_features] = label_dict[k]
        i += 1

    return all_features_divided_per_game, all_label_divided_per_game
'''


'''a, b = dataset_generator("/home/alberto/Scrivania/SoccerNet-code-master/data", "ResNET", )
print(len(a[0, 0, 0, :]))
print(a[0, 52, 12, :])
print(b.shape)
'''
