import cv2 as cv
import os
import csv
import gdown
import argparse
import numpy as np


ROOT_FOLDER = "faces"



# This is the data recording pipeline
def take_image(args):
    if args.folder is None:
        print("Please specify folder for data to be recorded into")
        exit()

    # create objects folder if it doesn't exist already
    if not os.path.exists(ROOT_FOLDER):
        os.mkdir(ROOT_FOLDER)

    # define the output folder with the chosen name (args)
    output_folder = os.path.join(ROOT_FOLDER, args.folder)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # URL of the XML file on Google Drive
    cascade_url = "https://drive.google.com/uc?id=1x4ejdgmTQMEtOMLwdNkKIHxkUYnSLabl"

    cascade_file = "haarcascade_frontalface_default.xml"

    # Check if the cascade file already exists locally
    if not os.path.exists(cascade_file):
        # If the file does not exist, download it from Google Drive
        print("Downloading the cascade file...")
        gdown.download(cascade_url, cascade_file)
    else:
        print("File already exists")

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + cascade_file)


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detect face
        face = face_cascade.detectMultiScale(gray, 1.3, 5)

        frame_visualized = frame.copy()

        for x, y, w, h in face:
            cv.rectangle(frame_visualized, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv.imshow("frame", frame_visualized)

        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # to only save every 30. frame
        if key == ord("y"):

            # when a face is detected
            if len(face) == 1:

                # Simulate a flash effect
                flash_frame = np.ones(frame.shape, dtype=np.uint8) * 255  # Create a white screen
                cv.imshow("frame", flash_frame)  # Show the white screen
                cv.waitKey(200)  # Wait for 200 milliseconds (simulate flash)

                # crop face and save it as png file
                for x, y, w, h in face:
                    x_inc = int(w*0.3)
                    y_inc = int(h*0.3)
                    # crop face region
                    cropped_face = frame[y-y_inc:y+h+y_inc, x-x_inc:x+w+x_inc]
                    
                    # save cropped face image
                    cv.imwrite(os.path.join(output_folder, "cropped_face.png"), cropped_face)
            cv.waitKey(500)
            break
            

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def transform_for_gan(args):
    # define the output folder with the chosen name (args)
    output_folder = os.path.join(ROOT_FOLDER, args.folder)
    input_img_path = os.path.join(output_folder, "cropped_face.png")

    # resize cropped face to 1024x1024 to fit our StyleGAN
    cropped_face = cv.imread(input_img_path)
    resized_face = cv.resize(cropped_face, (1024, 1024), interpolation=cv.INTER_CUBIC)
    cv.imwrite(os.path.join(output_folder, "transformed_face.png"), resized_face)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record face data from webcam")
    parser.add_argument("--folder", type=str, required=True, help="Folder name to save face images")
    args = parser.parse_args()

    take_image(args)
    transform_for_gan(args)