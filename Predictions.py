from keras.models import load_model

from Dataset_Detection import *
from display_video.graph_overlay import plot_prediction_over_video
from display_video.plt_func_animation import create_prediction_video
from metrics import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


# Funzione che genera una predizione al secondo, usando un modello precedentemente addestrato
def generate_predictions(model, test_games=None):
    if test_games is None:
        _, _, test_games = read_data()
    for match, label, game, feature_game_name in dataset_detection_generator(test_games,
                                                                             "/data/datasets/soccernet/SoccerNet-code/data",
                                                                             "ResNET"):
        if "1_" in feature_game_name:
            prediction_name = 'My_prediction_half1'
        else:
            prediction_name = 'My_prediction_half2'
        if not os.path.exists('Data'):
            os.mkdir('Data')
        directory_path = os.path.join('Data', game)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        destination_path = os.path.join(directory_path, prediction_name)
        predictions = model.predict(match)
        # scores = model.evaluate(match,label)
        np.save(destination_path, predictions)


# Funzione che genera due video: un video solo delle predizioni, l'altro con le predizioni sopra il video della partita
def generate_prediction_over_a_video(path_prediction, path_videostats_prediction, path_filename_match,
                                     path_overlay_video, only_stats=True,
                                     scaling_video_factor=4, scaling_stats_factor=0.7):
    create_prediction_video(path_prediction, path_videostats_prediction)
    if not only_stats:
        plot_prediction_over_video(path_filename_match, path_videostats_prediction, path_overlay_video,
                                   scaling_video_factor, scaling_stats_factor)


# Funzione che itera la funzione precedente
def generate_all_prediction_over_a_video(test_games=None, only_stats=True):
    if test_games is None:
        _, _, test_games = read_data()
    base_video = '/data/datasets/soccernet'
    base_output = 'Data'
    for game in test_games:
        prediction_folder_path = os.path.join(base_output, game)
        video_folder_path = os.path.join(base_video, game)
        prediction1_path = os.path.join(prediction_folder_path, 'My_prediction_half1.npy')
        video1_path = os.path.join(video_folder_path, '1.mkv')
        generate_prediction_over_a_video(prediction1_path,
                                         os.path.join(prediction_folder_path, 'video_stats_half1.mp4'),
                                         video1_path,
                                         os.path.join(prediction_folder_path, 'video_stats+match_half1.mp4'),
                                         only_stats)

        prediction2_path = os.path.join(prediction_folder_path, 'My_prediction_half2.npy')
        video2_path = os.path.join(video_folder_path, '2.mkv')
        generate_prediction_over_a_video(prediction2_path,
                                         os.path.join(prediction_folder_path, 'video_stats_half2.mp4'),
                                         video2_path,
                                         os.path.join(prediction_folder_path, 'video_stats+match_half2.mp4'),
                                         only_stats)


model = load_model('./Model/saved-model-34-0.2283-0.7823.h5',
                   custom_objects={'auprc': auprc, 'auprc0': auprc0, 'auprc1': auprc1, 'auprc2': auprc2,
                                   'auprc3': auprc3, 'auprc1to3': auprc1to3, 'f1m': f1m})

# Tests
_, _, test_games = read_data()
generate_predictions(model, test_games[0:35])
generate_all_prediction_over_a_video(test_games[0:35])
# generate_all_prediction_over_a_video(test_games[0:2], False)
