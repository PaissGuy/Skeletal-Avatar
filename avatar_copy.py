import sys
import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from pose_format import Pose
from pose_format.utils.holistic import load_holistic
from pose_format.utils.openpose import load_openpose_directory
from pose_format.pose_visualizer import PoseVisualizer

mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))]

def load_video_frames(cap: cv2.VideoCapture):
    """
    This is a generator that yield one frame at a time
    :param cap:
    :return:
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

def get_filtered_pose_landmarks():
    # Define the pose landmarks to keep (excluding certain points)
    pose_landmarks = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE',
                      'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER',
                      'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY',
                      'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP',
                      'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL',
                      'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    # Define the points to exclude (for example, hips and knees)
    exclude_points = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22}  # Left_Pinky, Right_Pinky, Left_Index, Right_Index, Left_Thumb, Right_Thumb
    # Generate list of points to keep (0-32 excluding exclude_points)
    filtered = [i for i in range(33) if i not in exclude_points]
    return [pose_landmarks[i] for i in filtered]

def get_filtered_hand_points():
    hands_landmarks = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
    # Define the palm points to exclude (these form the triangle)
    palm_points = {0, 1, 2, 5, 9, 13, 17, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20}
    # # Generate list of points to keep (0-20 excluding palm points)
    filtered = [i for i in range(21) if i not in palm_points]
    # print(f"Filtered hand points (kept): {filtered}")  # Debug: Print kept points
    return [hands_landmarks[i] for i in filtered]


def pose_estimate(video_path, output_path, lib='mediapipe', reduce=False):
    """
        Converts video to pose object and returns it.
    :param video_path: The video absolute path
    :param output_path:
    :param lib:
    :param reduce:
    :return:
    """
    # Load video frames
    print('Load video ...')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = (cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video metadata: fps={fps}, width={width}, height={height}, frames={total_frames}")

    frames = load_video_frames(cap)
    video_metadata = dict(fps=fps, width=width, height=height)

    # Perform pose estimation
    print('Estimating pose ...')

    if lib == 'mediapipe':
        pose = load_holistic(frames, fps=fps, width=width, height=height, progress=True,
                         additional_holistic_config={'model_complexity': 2, 'refine_face_landmarks': True}) # Returns Pose Object after detecting and tracking multiple key points
    #pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])

    if reduce:
        filtered_pose_landmarks = get_filtered_pose_landmarks()
        pose = pose.get_components(
            ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
            {
                "FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS,
                "POSE_LANDMARKS": filtered_pose_landmarks
            }
        )
        # pose = pose.get_components(
        #     [ "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
        #     {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS}
        # )
    elif lib == 'openpose':
        pose = load_openpose_directory(video_path.replace('.mp4', '.openpose'), fps=fps, width=width, height=height)

    print('Points:', pose.body.data.shape)

    # Write output
    print('Writing pose file ...', output_path)
    with open(output_path, "wb") as f:
        pose.write(f)

def pose_visualize(video_path, pose_path, overlay=False):
    print('Reading pose file ...')
    with open(pose_path, "rb") as f:
        buffer = f.read()
        pose = Pose.read(buffer)

    v = PoseVisualizer(pose, thickness=2)

    if overlay:
        out_file = f"{pose_path}.overlay.mp4"
        print("Saving overlay visualization:", out_file)
        v.save_video(out_file, v.draw_on_video(video_path))
    else:
        out_file = f"{pose_path}.mp4"
        print("Saving skeleton-only visualization:", out_file)
        v.save_video(out_file, v.draw())

def find_video(filename=None, root=None):
    """
    Returns the absolute path of the video
    :param filename:
    :param root:
    :return:
    """

    root = root or os.getcwd()
    if filename:
        # Try exact path
        if os.path.isabs(filename) and os.path.exists(filename):
            return os.path.abspath(filename)
        rel = os.path.join(root, filename)
        if os.path.exists(rel):
            return os.path.abspath(rel)
    # If no filename or not found, search for mp4/mov/avi
    for ext in ('.mp4', '.mov', '.avi', '.mkv'):
        matches = glob.glob(os.path.join(root, '**', f'*{ext}'), recursive=True)
        if matches:
            return os.path.abspath(matches[0])
    return None

if __name__ == "__main__":
    ROOT = os.getcwd() #Get current working directory

    
    video_filename = "class_1.avi"   # put your video filename here

    video_path = find_video(video_filename, ROOT)
    if not video_path:
        print(f" Video not found. Put '{video_filename}' in {ROOT} or update video_filename.")
        sys.exit(1)

    print(" Using video:", video_path)

    # Make output folder next to video
    out_dir = os.path.join(os.path.dirname(video_path), "output")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, os.path.splitext(os.path.basename(video_path))[0] + ".mediapipe.pose")

    lib = 'mediapipe'
    visualize = True
    reduce = True

    pose_estimate(video_path, output_path, reduce=reduce, lib=lib)
    if visualize:
        pose_visualize(video_path, output_path, overlay=True)
        pose_visualize(video_path, output_path, overlay=False)

    print("Done.")