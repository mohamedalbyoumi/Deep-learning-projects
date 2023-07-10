import cv2
import numpy as np
#from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
def getPrediction(video_path, output_path, apply_mask=False):
    # Load the trained model
    my_model =  tfa.models.segmentation.unet.Unet()

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

    # Return the path to the output video
    return output_path
