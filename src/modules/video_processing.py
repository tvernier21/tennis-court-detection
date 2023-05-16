import cv2
import os

def video_to_frames(video_path, video_file, output_dir):
    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
    # Open the video file
    video = cv2.VideoCapture(video_path)
    # Initialize a counter for the frame number
    frame_num = 0

    while True:
        # Read the next frame
        ret, frame = video.read()
        # If the frame was not successfully read, then we have reached the end of the video
        if not ret:
            break
        # Construct the output file path
        output_file = os.path.join(output_dir, f"{video_file}_{frame_num}.png")

        # Normalize and write the frame to an image file
        resized_frame = cv2.resize(frame, (256, 256))
        cv2.imwrite(output_file, resized_frame)

        # Increment the frame number
        frame_num += 1
    video.release()


if __name__ == '__main__':
    # process every video in the videos folder
    for video in os.listdir('data/videos'):
        print(f'Processing {video}...')
        video_to_frames(f'data/videos/{video}', video.split(".")[0], 'data/images')