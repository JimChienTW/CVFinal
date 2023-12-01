import os
import argparse
import sys


import cv2
import numpy as np
from cprint import *

import face_recognition
from utils.face_utils import change_faces


from deepface import DeepFace

def load_known_face(path):
    image = face_recognition.load_image_file(path)
    # image_encoding = face_recognition.face_encodings(image)[0]
    image_encoding = face_recognition.face_encodings(image)
    image_name = str(path).split('/')[-1].split('.')[0]
    return image_encoding[0], image_name

def load_known_faces(known_faces_dir='face_base'):
    known_face_encodings = []
    known_face_names = []

    files = [file for file in os.listdir(known_faces_dir) if os.path.isfile(os.path.join(known_faces_dir, file)) and file.lower().endswith('.jpg')]
    print(files)
    for file in files:
        image_encoding, image_name = load_known_face(os.path.join(known_faces_dir, file))
        known_face_encodings.append(image_encoding)
        known_face_names.append(image_name)
    return known_face_encodings, known_face_names, len(known_face_names)


def add_faces(known_face_encodings, known_face_names, known_face_num, face_locations, face_names, frame, opt):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name != 'Unknown':
            continue
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4


        # scale_ratio = 2
        # center_x = (left + right) / 2
        # center_y = (top + bottom) / 2
        # print(center_x, center_y)
        # w = (right - left)
        # h = (bottom - top)
        # new_w, new_h = w * scale_ratio, h * scale_ratio
        # top, bottom = int(center_y - 0.5 * new_h), int(center_y + 0.5 * new_h)
        # left, right = int(center_y - 0.5 * new_w), int(center_x + 0.5 * new_w)
        # print(top, bottom, left, right)
        cv2.imwrite('../deepface/0.jpg', frame)






        face_image_new_raw = frame[top:bottom, left:right]
        face_image_new_raw = cv2.resize(face_image_new_raw, (opt.face_img_size, opt.face_img_size))
        cv2.imwrite(os.path.join(opt.known_face_dir, f'{known_face_num}.jpg'), face_image_new_raw)
        face_image_new_encoding, face_image_new_name = load_known_face(os.path.join(opt.known_face_dir, f'{known_face_num}.jpg'))
        known_face_encodings.append(face_image_new_encoding)
        known_face_names.append(face_image_new_name)            
        known_face_num += 1

    return known_face_encodings, known_face_names, known_face_num



def main(opt):
    cprint.info('Game started')        
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # load know face from 
    if opt.known_face_dir is not None:
        os.makedirs(opt.known_face_dir, exist_ok=True)
        known_face_encodings, known_face_names, known_face_num = load_known_faces(opt.known_face_dir)
        cprint.info(known_face_names)

    # Initialize variable of face
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    # Webcam video stream
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
            
                # use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    # smaller threshold to avoid confusion
                    if face_distances[best_match_index] < 0.5:
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                face_names.append(name)

        process_this_frame = not process_this_frame

        # Add new player to known face        
        if cv2.waitKey(1) & 0xFF == ord('r'):
            known_face_encodings, known_face_names, known_face_num = add_faces(known_face_encodings, known_face_names, known_face_num, face_locations, face_names, frame, opt)


        # change face
        if cv2.waitKey(1) & 0xFF == ord('m'):
            print('asdfasfasdfsfafsdf')
            # detect the facial key point, then save and update
            change_faces(face_names, opt)
            # read the delaunay triangle 

            # update the morphed face

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            
            demography = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, )
            emotion_info = demography[0]['emotion']
            emo = max(emotion_info, key=emotion_info.get)


            if name == 'Unknown':
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name + '\n' + emo, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            else:
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


            if os.path.exists(os.path.join(opt.changed_face_dir, name + '.jpg')):
                replaced_image = cv2.imread(os.path.join(opt.changed_face_dir, name + '.jpg'))
                replaced_image = cv2.resize(replaced_image, (right-left, bottom-top))
                frame[top:bottom, left:right] = replaced_image
        
        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--known_face_dir', type=str, default='face_base')
    parser.add_argument('--target_face_dir', type=str, default='target_face')
    parser.add_argument('--changed_face_dir', type=str, default='changed_face')
    parser.add_argument('--keypoints_dir', type=str, default='keypoints')
    parser.add_argument('--tri_dir', type=str, default='tri')
    parser.add_argument('--face_img_size', type=int, default=640)
    opt = parser.parse_args()
    return opt



if __name__ == '__main__':
    opts = opt_parser()
    main(opts)
