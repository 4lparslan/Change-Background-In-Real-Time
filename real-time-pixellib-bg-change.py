import pixellib
from pixellib.tune_bg import alter_bg
import cv2

cam = cv2.VideoCapture(0)

change_bg = alter_bg(model_type = "pb")
change_bg.load_pascalvoc_model("../xception_pascalvoc.pb")
change_bg.blur_camera(cam, frames_per_second=30, extreme = True, show_frames=True, frame_name="frame", detect = "person", output_video_name="blured_video.mp4")
