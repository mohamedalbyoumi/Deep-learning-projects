import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the trained model
my_model = load_model("model/model.h5")

# Define the video operations
VIDEO_OPERATIONS = {
    'mask': {'apply_mask': True, 'output_folder': 'output\masked_videos'},
    'threshold': {'apply_mask': False, 'output_folder': 'output\thresholded_videos'},
}

def get_prediction(video_path, output_path, apply_mask=False):
    # Define the video writer
    fps = 30  # Frames per second
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (512, 512))

    # Open the input video file
    video = cv2.VideoCapture(video_path)

    # Read and process each frame
    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Resize the frame to the desired size
        frame = cv2.resize(frame, (512, 512))

        # Preprocess the frame
        # Apply any necessary preprocessing steps here

        # Make predictions on the frame using the model
        # You may need to adapt this code based on your specific model's input requirements
        processed_frame = frame / 255.0  # Normalize the pixel values
        prediction = my_model.predict(np.expand_dims(processed_frame, axis=0))

        # Apply post-processing to the prediction
        # This step depends on the output format of your model
        masked_frame = prediction[0] * 255.0  # De-normalize the pixel values

        # Convert the masked frame to the correct data type
        masked_frame = masked_frame.astype(np.uint8)

        if apply_mask:
            # Apply the mask to the original frame
            overlay = frame * (masked_frame / 255.0)

            # Write the overlay frame to the output video
            output_video.write(overlay)
        else:
            # Write the masked frame to the output video
            output_video.write(masked_frame)

    # Release the video capture and writer
    video.release()
    output_video.release()

    return None


def main():
    st.title("Video Processing")

    # File upload
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4'])

    if uploaded_file:
        operation = st.selectbox("Select Operation", list(VIDEO_OPERATIONS.keys()))

        if st.button("Process"):
            # Save the uploaded video to a temporary file
            video_path = r"input\temp_video.mp4"
            with open(video_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            operation_params = VIDEO_OPERATIONS[operation]
            apply_mask = operation_params['apply_mask']
            output_folder = operation_params['output_folder']

            output_path = os.path.join(output_folder, operation + '_' + video_path)

            # Call the get_prediction function passing the video path, output path, and apply_mask parameter
            get_prediction(video_path, output_path, apply_mask)
            st.success('Operation on video done successfully.')

            # Display the input and operated videos
            st.subheader("Input Video:")
            st.video(video_path)

            st.subheader("Operated Video:")
            st.video(output_path)


if __name__ == "__main__":
    main()
