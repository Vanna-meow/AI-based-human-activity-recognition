
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# Dataset location
DATASET_DIR = "dataset"
OUTPUT_FILE = "features.csv"

# Five activities
ACTIVITIES = [
    "walking",
    "running",
    "standing",
    "sitting",
    "waving"
]

VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")

mp_pose = mp.solutions.pose


def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_features = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return None

        frame_indices = np.linspace(
            0, total_frames - 1,
            min(20, total_frames),
            dtype=int
        )

        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results = pose.process(frame)

            if results.pose_landmarks:
                landmarks = []

                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ])

                frame_features.append(landmarks)

    cap.release()

    if not frame_features:
        return None

    features = np.array(frame_features)

    # Mean and standard deviation
    mean_features = np.mean(features, axis=0)
    std_features = np.std(features, axis=0)

    return np.concatenate([
        mean_features,
        std_features
    ])


def main():
    all_data = []

    for activity in ACTIVITIES:
        activity_dir = os.path.join(
            DATASET_DIR, activity
        )

        if not os.path.isdir(activity_dir):
            print("Missing folder:", activity_dir)
            continue

        videos = [
            f for f in os.listdir(activity_dir)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ]

        print(activity, "videos:", len(videos))

        for video in videos:
            video_path = os.path.join(
                activity_dir, video
            )

            try:
                features = extract_video_features(
                    video_path
                )

                if features is None:
                    print("No pose detected:", video)
                    continue

                row = features.tolist()
                row.append(activity)

                all_data.append(row)

                print("Processed:", video)

            except Exception as error:
                print("Error:", video, error)

    if not all_data:
        print("No features extracted. Check dataset folders.")
        return

    columns = (
        [f"feature_{i}" for i in range(264)]
        + ["activity"]
    )

    df = pd.DataFrame(
        all_data,
        columns=columns
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nFeature extraction completed!")
    print("Total videos:", len(df))
    print(df["activity"].value_counts())
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

