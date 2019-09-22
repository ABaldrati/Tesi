from Dataset import *
import gc


# Funzione che genera uno sliding windows di features
def dataset_detection_generator_complete(list_games, path_data, feature_type, chunk_size=60, PCA=True):
    FEATURE_PER_SECOND = 2

    for game in list_games:
        game_path = os.path.join(path_data, game)
        for feature_game_name in sorted(os.listdir(game_path)):
            if feature_type in feature_game_name and (
                    (PCA and "PCA" in feature_game_name) or (not PCA and "PCA" not in feature_game_name)):
                feature_game_path = os.path.join(game_path, feature_game_name)

                current_features_rough = np.load(feature_game_path)

                strides = current_features_rough.strides
                nb_frames = current_features_rough.shape[0]
                size_feature = current_features_rough.shape[1]
                sliding_window_seconds = chunk_size
                sliding_window_frame = sliding_window_seconds * FEATURE_PER_SECOND

                current_features_rough = np.append(
                    [current_features_rough[0, :] * 0] * sliding_window_seconds, current_features_rough,
                    axis=0)
                current_features_rough = np.append(current_features_rough,
                                                   [current_features_rough[0, :] * 0] * sliding_window_seconds, axis=0)
                current_features = np.lib.stride_tricks.as_strided(current_features_rough, shape=(
                    int(nb_frames / 2), sliding_window_frame, size_feature),
                                                                   strides=(strides[0] * 2, strides[0], strides[1]))
                label_game_path = os.path.join(game_path, "Labels.json")
                with open(label_game_path) as labelFile:
                    jsonLabel = json.loads(labelFile.read())

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
                        around_value = min(time, label.shape[0] - 1)
                        label[(around_value - int(sliding_window_seconds / 2)):(
                                around_value + int(sliding_window_seconds / 2)), 0] = 0
                        label[(around_value - int(sliding_window_seconds / 2)):(
                                around_value + int(sliding_window_seconds / 2)), curr_label] = 1

                yield current_features, label, game, feature_game_name
