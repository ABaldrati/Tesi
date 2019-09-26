import json
import os

import numpy as np
#from keras.models import load_model
from tqdm import tqdm

from Dataset import read_data
from Highlights import spot_actions
from Predictions import generate_predictions
from metrics import *


def count_single_events(label_path, feature_game_name, type_of_action):
    with open(label_path) as labelFile:
        jsonLabel = json.loads(labelFile.read())

    num_single_events = 0
    last_time = -100  # inizialization
    for event in jsonLabel["annotations"]:
        if type_of_action in event['label']:
            time_half = int(event["gameTime"][0])
            time_minute = int(event["gameTime"][-5:-3])
            time_second = int(event["gameTime"][-2:])

            if ("1_" in feature_game_name and time_half == 1) or (
                    "2_" in feature_game_name and time_half == 2):
                time = time_minute * 60 + time_second
                if (time - last_time) > 30 and time != 0:
                    num_single_events += 1
                last_time = time

    return num_single_events


# Funziona che calcola risultati del task di spotting con 30 secondi di tolleranza
def calculate_spotting_results(list_games, path_data, feature_type, chunk_size=60, PCA=True):
    true_positives_card = 0
    false_positives_card = 0
    # true_negatives_card = 0
    false_negatives_card = 0

    true_positives_subs = 0
    false_positives_subs = 0
    # true_negatives_subs = 0
    false_negatives_subs = 0

    true_positives_soccer = 0
    false_positives_soccer = 0
    # true_negatives_soccer = 0
    false_negatives_soccer = 0

    for game in tqdm(list_games):
        game_path = os.path.join(path_data, game)
        for feature_game_name in sorted(os.listdir(game_path)):
            if feature_type in feature_game_name and (
                    (PCA and "PCA" in feature_game_name) or (not PCA and "PCA" not in feature_game_name)):
                feature_game_path = os.path.join(game_path, feature_game_name)

                current_features_rough = np.load(feature_game_path)
                nb_frames = current_features_rough.shape[0]
                sliding_window_seconds = chunk_size

                label_game_path = os.path.join(game_path, "Labels.json")
                with open(label_game_path) as labelFile:
                    jsonLabel = json.loads(labelFile.read())
                label = np.zeros((int(nb_frames / 2), 4), dtype=np.uint8)
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
                        around_value = min(time, label.shape[0] - 1)
                        label[(around_value - int(sliding_window_seconds / 2)):(
                                around_value + int(sliding_window_seconds / 2)), 0] = 0
                        label[(around_value - int(sliding_window_seconds / 2)):(
                                around_value + int(sliding_window_seconds / 2)), curr_label] = 1

                if "1_" in feature_game_name:
                    predictions = np.load(os.path.join('Data', game, 'My_prediction_half1.npy'))
                else:
                    predictions = np.load(os.path.join('Data', game, 'My_prediction_half2.npy'))

                predicted_card = spot_actions(predictions, 'card')
                predicted_subs = spot_actions(predictions, 'subs')
                predicted_soccer = spot_actions(predictions, 'soccer')

                true_positives_card_match = 0
                for predicted_action in predicted_card:
                    if label[int((predicted_action.start_second + predicted_action.end_second) / 2)][1] == 1:
                        true_positives_card += 1
                        true_positives_card_match += 1
                    else:
                        false_positives_card += 1
                counter_events_card = count_single_events(label_game_path, feature_game_name, 'card')
                false_negatives_card += counter_events_card - true_positives_card_match

                true_positives_subs_match = 0
                for predicted_action in predicted_subs:
                    if label[int((predicted_action.start_second + predicted_action.end_second) / 2)][2] == 1:
                        true_positives_subs += 1
                        true_positives_subs_match += 1
                    else:
                        false_positives_subs += 1
                counter_events_subs = count_single_events(label_game_path, feature_game_name, 'subs')
                false_negatives_subs += counter_events_subs - true_positives_subs_match

                true_positives_soccer_match = 0
                for predicted_action in predicted_soccer:
                    if label[int((predicted_action.start_second + predicted_action.end_second) / 2)][3] == 1:
                        true_positives_soccer += 1
                        true_positives_soccer_match += 1
                    else:
                        false_positives_soccer += 1
                counter_events_soccer = count_single_events(label_game_path, feature_game_name, 'soccer')
                false_negatives_soccer += counter_events_soccer - true_positives_soccer_match

    precision_card = true_positives_card / (true_positives_card + false_positives_card)
    recall_card = true_positives_card / (true_positives_card + false_negatives_card)
    f1_card = 2 * (precision_card * recall_card) / (precision_card + recall_card)

    precision_subs = true_positives_subs / (true_positives_subs + false_positives_subs)
    recall_subs = true_positives_subs / (true_positives_subs + false_negatives_subs)
    f1_subs = 2 * (precision_subs * recall_subs) / (precision_subs + recall_subs)

    precision_soccer = true_positives_soccer / (true_positives_soccer + false_positives_soccer)
    recall_soccer = true_positives_soccer / (true_positives_soccer + false_negatives_soccer)
    f1_soccer = 2 * (precision_soccer * recall_soccer) / (precision_soccer + recall_soccer)

    print(true_positives_card, false_positives_card, false_negatives_card, precision_card, recall_card, f1_card)
    print(true_positives_subs, false_positives_subs, false_negatives_subs, precision_subs, recall_subs, f1_subs)
    print(true_positives_soccer, false_positives_soccer, false_negatives_soccer, precision_soccer, recall_soccer,
          f1_soccer)

    print(f1_card, f1_subs, f1_soccer)

    return f1_card, f1_subs, f1_soccer


def main():
    #model = load_model('./Model/saved-model-34-0.2283-0.7823.h5',
                       #custom_objects={'auprc': auprc, 'auprc0': auprc0, 'auprc1': auprc1, 'auprc2': auprc2,
                                       #'auprc3': auprc3, 'auprc1to3': auprc1to3, 'f1m': f1m})

    #generate_predictions(model)
    _, _, test_games = read_data()
    calculate_spotting_results(test_games, '/data/datasets/soccernet/SoccerNet-code/data', 'ResNET')


if __name__ == '__main__':
    main()
