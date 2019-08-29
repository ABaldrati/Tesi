import os
from Dataset import read_data
import numpy as np

from display_video.cut_video import cut_video


class Action:
    def __init__(self, start, end, type_of_action):
        self.start_second = start
        self.end_second = end
        self.type_of_action = type_of_action


# Funzione che restituisce un array di `Action` del tipo desiderato
def spot_actions(prediction_file, type_of_action, toll=0.9, min_duration=10, toll_seconds=3):
    if 'card' in type_of_action:
        index = 1
    elif 'subs' in type_of_action:
        index = 2
    elif 'soccer' in type_of_action:
        index = 3
    else:
        index = 5  # indexerror

    in_action = False

    actions = []

    for second in range(len(prediction_file)):
        if prediction_file[second][index] > toll:
            counter_toll = 0
            if in_action is False:
                start_action = second
                in_action = True
        else:
            if in_action is True:
                if counter_toll == 0:
                    end_action = second
                if counter_toll < toll_seconds:
                    counter_toll += 1
                else:
                    in_action = False
                    if end_action - start_action > min_duration:
                        actions.append(Action(start_action, end_action, type_of_action))
    return actions


# Funzione che genera gli Highlights di una partita
def generate_single_highlights(game):
    base_video = '/data/datasets/soccernet'
    base_highlight = os.path.join('Data', game, 'Highlights')
    if not os.path.exists(base_highlight):
        os.makedirs(base_highlight)

    # HALF1
    prediction_half1 = np.load(os.path.join('Data', game, 'My_prediction_half1.npy'))
    video_path = os.path.join(base_video, game, '1.mkv')

    actions_card = spot_actions(prediction_half1, 'card')
    actions_subs = spot_actions(prediction_half1, 'subs')
    actions_soccer = spot_actions(prediction_half1, 'soccer')

    for action in actions_card:
        output_name = 'Half1 ' + str(action.start_second // 60) + ':' + str(action.start_second % 60) + ' - ' + str(
            action.end_second // 60) + ':' + str(action.end_second % 60) + ' Card.mp4'
        cut_video(video_path, os.path.join(base_highlight, output_name), action.start_second, action.end_second)

    for action in actions_subs:
        output_name = 'Half1 ' + str(action.start_second // 60) + ':' + str(action.start_second % 60) + ' - ' + str(
            action.end_second // 60) + ':' + str(action.end_second % 60) + ' Subs.mp4'
        cut_video(video_path, os.path.join(base_highlight, output_name), action.start_second, action.end_second)

    for action in actions_soccer:
        output_name = 'Half1 ' + str(action.start_second // 60) + ':' + str(action.start_second % 60) + ' - ' + str(
            action.end_second // 60) + ':' + str(action.end_second % 60) + ' Goal.mp4'
        cut_video(video_path, os.path.join(base_highlight, output_name), action.start_second, action.end_second)

    # HALF2
    prediction_half2 = np.load(os.path.join('Data', game, 'My_prediction_half2.npy'))
    video_path = os.path.join(base_video, game, '2.mkv')

    actions_card = spot_actions(prediction_half2, 'card')
    actions_subs = spot_actions(prediction_half2, 'subs')
    actions_soccer = spot_actions(prediction_half2, 'soccer')

    for action in actions_card:
        output_name = 'Half2 ' + str(action.start_second // 60) + ':' + str(action.start_second % 60) + ' - ' + str(
            action.end_second // 60) + ':' + str(action.end_second % 60) + ' Card.mp4'
        cut_video(video_path, os.path.join(base_highlight, output_name), action.start_second, action.end_second)

    for action in actions_subs:
        output_name = 'Half2 ' + str(action.start_second // 60) + ':' + str(action.start_second % 60) + ' - ' + str(
            action.end_second // 60) + ':' + str(action.end_second % 60) + ' Subs.mp4'
        cut_video(video_path, os.path.join(base_highlight, output_name), action.start_second, action.end_second)

    for action in actions_soccer:
        output_name = 'Half2 ' + str(action.start_second // 60) + ':' + str(action.start_second % 60) + ' - ' + str(
            action.end_second // 60) + ':' + str(action.end_second % 60) + ' Goal.mp4'
        cut_video(video_path, os.path.join(base_highlight, output_name), action.start_second, action.end_second)


def generate_all_highlights(test_games=None):
    if test_games is None:
        _, _, test_games = read_data()
    for game in test_games:
        generate_single_highlights(game)


_, _, test_games = read_data()
generate_all_highlights(test_games[0:10])
