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


'''def count_single_class(list_games, path_data, feature_type, feature_per_chunk, PCA):
    label_dict = {}
    chunk_size = int(feature_per_chunk / 2)
    for game in tqdm(list_games):
        game_path = os.path.join(path_data, game)
        for feature_game_name in os.listdir(game_path):
            if feature_type in feature_game_name and (
                    (PCA and "PCA" in feature_game_name) or (not PCA and "PCA" not in feature_game_name)):
                feature_game_path = os.path.join(game_path, feature_game_name)

                current_features_rough = np.load(feature_game_path)

                current_features = np.zeros((int(math.ceil(current_features_rough.shape[0] / feature_per_chunk)),
                                             feature_per_chunk, current_features_rough.shape[1]))

                for i in range(current_features_rough.shape[0]):
                    current_features[i // feature_per_chunk, i % feature_per_chunk, :] = current_features_rough[i]

                label = np.zeros((current_features.shape[0], 4), dtype=int)
                label[:, 0] = 1
                label_game_path = os.path.join(game_path, "Labels.json")
                with open(label_game_path) as labelFile:
                    jsonLabel = json.loads(labelFile.read())

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

                label_dict[feature_game_path] = label
    n_back = 0
    n_card = 0
    n_subs = 0
    n_goal = 0
    for key, label in label_dict.items():
        for i in range(label.shape[0]):
            if label[i, 0] == 1:
                n_back += 1
            if label[i, 1] == 1:
                n_card += 1
            if label[i, 2] == 1:
                n_subs += 1
            if label[i, 3] == 1:
                n_goal += 1
    return n_back, n_card, n_subs, n_goal
'''

'''def find_tensor_dim(list_games, path_data, feature_type, feature_per_chunk, PCA):
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
'''


def dataset_generator(list_games, path_data, feature_type, batch_size, data_aug=True, chunk_size=60, PCA=True):
    FEATURE_PER_SECOND = 2
    feature_per_chunk = FEATURE_PER_SECOND * chunk_size

    # max_chunk_feature_per_match, n_features = find_tensor_dim(list_games, path_data, feature_type, feature_per_chunk,
    # PCA)

    if PCA:
        n_features = 512
    elif "ResNET" in feature_type:
        n_features = 2048
    elif "C3D" in feature_type:
        n_features = 4096
    elif "I3D" in feature_type:
        n_features = 1024

    max_chunk_feature = 0
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

                        if ("1_" in feature_game_name and time_half == 1) or (
                                "2_" in feature_game_name and time_half == 2) and (data_aug is True):

                            t = time_minute * 60 + time_second
                            t_ini = t - int(chunk_size * 0.667 / 2.0)
                            f_ini = t_ini * FEATURE_PER_SECOND
                            t_end = t + int(chunk_size * 0.667 / 2.0)
                            f_end = t_end * FEATURE_PER_SECOND

                            for f in range(f_ini, f_end, 1):
                                if (f + chunk_size < len(current_features_rough)) and (f - chunk_size > 0):
                                    num_aug += 1

                    current_features = np.zeros(
                        (int(math.ceil(current_features_rough.shape[0] / feature_per_chunk) + num_aug),
                         feature_per_chunk, current_features_rough.shape[1]))

                    for i in range(current_features_rough.shape[0]):
                        current_features[i // feature_per_chunk, i % feature_per_chunk, :] = current_features_rough[i]
                        next_feat_aug = i // feature_per_chunk + 1

                    feature_dict[feature_game_path] = current_features

                    label = np.zeros((current_features.shape[0], 4), dtype=int)
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
                                t_ini = t - int(chunk_size * 0.667 / 2.0)
                                f_ini = t_ini * FEATURE_PER_SECOND
                                t_end = t + int(chunk_size * 0.667 / 2.0)
                                f_end = t_end * FEATURE_PER_SECOND

                                for f in range(f_ini, f_end, 1):
                                    if (f + chunk_size < len(current_features_rough)) and (f - chunk_size > 0):
                                        extra_feat = current_features_rough[f - chunk_size:f + chunk_size, :]
                                        current_features[next_feat_aug] = extra_feat

                                        label[next_feat_aug, 0] = 0
                                        label[next_feat_aug, curr_label] = 1
                                        next_feat_aug += 1

                    max_chunk_feature = max(max_chunk_feature, current_features.shape[0])
                    label_dict[feature_game_path] = label

                    if n_sample % batch_size == 0:
                        features_per_game = np.zeros((len(feature_dict), max_chunk_feature, feature_per_chunk,
                                                      n_features))
                        labels_per_game = np.zeros((len(label_dict), max_chunk_feature, 4), dtype=int)
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
                        max_chunk_feature = 0


def generators(path_data, feature_type, batch_size, chunk_size=60, PCA=True):
    list_games = read_shuffle_data(path_data)
    train_games = list_games[0:300]
    validation_games = list_games[300:400]
    test_games = list_games[400:500]
    train_gen = dataset_generator(train_games, path_data, feature_type, batch_size, True, chunk_size, PCA)
    validation_gen = dataset_generator(validation_games, path_data, feature_type, batch_size, False, chunk_size, PCA)
    test_gen = dataset_generator(test_games, path_data, feature_type, batch_size, False, chunk_size, PCA)
    return train_gen, validation_gen, test_gen


'''list_games = read_shuffle_data("/home/alberto/Scrivania/SoccerNet-code-master/data")
a,b,c,d = count_single_class(list_games, "/home/alberto/Scrivania/SoccerNet-code-master/data", "ResNET", 120, True)
print(a)
print(b)
print(c)
print(d)'''
