# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
from moviepy.config import get_setting
from moviepy.tools import subprocess_call


# Funzione che estrae una porzione di video
def ffmpeg_extract_subclip(filename, t1, t2, targetname=None):
    """ Makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
    name, ext = os.path.splitext(filename)
    if not targetname:
        T1, T2 = [int(1000 * t) for t in [t1, t2]]
        targetname = "%sSUB%d_%d.%s" % (name, T1, T2, ext)

    cmd = [get_setting("FFMPEG_BINARY"), "-y",
           "-ss", "%0.2f" % t1,
           "-i", filename,
           "-t", "%0.2f" % (t2 - t1),
           "-vcodec", "copy", "-acodec", "copy", targetname]

    subprocess_call(cmd)


def cut_video(input_path, output_path, start_second, end_second):
    ffmpeg_extract_subclip(input_path, start_second, end_second, targetname=output_path)
