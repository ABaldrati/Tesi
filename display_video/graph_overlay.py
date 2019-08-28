import cv2
import numpy as np
import imutils


def plot_prediction_over_video(path_filename_match, path_filename_stats, path_output_video, scaling_video_factor=4,
                               scaling_stats_factor=0.7):
    cap_match = cv2.VideoCapture(path_filename_match, 0)
    cap_stats = cv2.VideoCapture(path_filename_stats, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    i = 0
    fps = cap_match.get(cv2.CAP_PROP_FPS)
    while cap_match.isOpened() and cap_stats.isOpened():
        ret_match, frame_match = cap_match.read()
        if i % fps == 0:
            ret_stats, frame_stats = cap_stats.read()
        i += 1

        if writer is None:
            (h, w) = frame_match.shape[:2]
            (hs, ws) = frame_stats.shape[:2]
            writer = cv2.VideoWriter(path_output_video, fourcc, fps,
                                     (int(w * scaling_video_factor), int(h * scaling_video_factor)), True)
        output = np.zeros((int(h * scaling_video_factor), int(w * scaling_video_factor), 3), dtype='uint8')
        if (frame_match is None) or (frame_stats is None):
            break
        frame_match = imutils.resize(frame_match, width=int(w * scaling_video_factor),
                                     height=int(h * scaling_video_factor))
        frame_stats = imutils.resize(frame_stats, height=int(hs * scaling_stats_factor),
                                     width=int(ws * scaling_stats_factor))
        small_frame_match = frame_match[0:int(hs * scaling_stats_factor), int(-ws * scaling_stats_factor):]
        weighted_frame_stats = cv2.addWeighted(small_frame_match, 0.4, frame_stats, 0.6, 0)

        output[:, :] = frame_match
        output[0:int(hs * scaling_stats_factor), int(-ws * scaling_stats_factor):] = weighted_frame_stats
        writer.write(output)

    # cv2.destroyAllWindows()
    cap_stats.release()
    cap_match.release()
    writer.release()
