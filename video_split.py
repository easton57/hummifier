""" File for splitting large video files into smaller trainable frames """
import cv2

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def split_video(file_name, increment, vid_num):
    # Get our length in seconds
    video_length = VideoFileClip(file_name).duration
    curr_time = 0
    times = []

    while curr_time < video_length:
        if curr_time + increment > video_length:
            times.append([curr_time, video_length])
        else:
            times.append([curr_time, curr_time + increment])

        curr_time += increment

    for time in times:
        start_time = int(time[0])
        end_time = int(time[1])
        ffmpeg_extract_subclip(file_name,
                               start_time,
                               end_time,
                               targetname=f"train_video/outside_window_{vid_num}_{time[0]}-{time[1]}.mp4")


def video_to_frames(file_name):
    # Get our length in seconds
    capture = cv2.VideoCapture(file_name)
    frame_nr = 0

    while (True):
        # Get the frame
        success, frame = capture.read()

        if success:
            cv2.imwrite(f'unsorted_photos/frame_{frame_nr}.jpg', frame)
        else:
            break

        frame_nr += 1

    capture.release()
