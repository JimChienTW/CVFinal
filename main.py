import cv2
import pygame
from pygame.locals import *
import numpy as np
import argparse
import face_recognition
from deepface import DeepFace
from cprint import *
import os
import random
from collections import deque
from collections import Counter
import time
from utils.face_utils import *

# Initialize Pygame
pygame.init()

# Set up the Pygame window
ORIG_WINDOW_WIDTH, ORIG_WINDOW_HEIGHT = 1920, 1080
WINDOW_WIDTH, WINDOW_HEIGHT = 1440, 810
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Video Stream Player")

# OpenCV video capture from webcam (change the index if you have multiple cameras)
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera

# Check if the video capture is successful
if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    pygame.quit()
    exit()

# Get the video properties
fps = 30  # Assuming a standard webcam provides 30 frames per second

# Set up Pygame clock
clock = pygame.time.Clock()


# define colours
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Define size of displayed morphed image
DISPLAYED_IMG_SIZE = 200
DISPLAYED_IMG_X_1, DISPLAYED_IMG_Y_1 = 15, 15
DISPLAYED_IMG_X_2, DISPLAYED_IMG_Y_2 = WINDOW_WIDTH - DISPLAYED_IMG_X_1 - DISPLAYED_IMG_SIZE, 15

# Define config of health bar
INTERVAL = 20
HEALTH_BAR_X_1, HEALTH_BAR_Y_1 = DISPLAYED_IMG_X_1 + DISPLAYED_IMG_SIZE + INTERVAL, 15
HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT = 200, 30
HEALTH_BAR_X_2, HEALTH_BAR_Y_2 = WINDOW_WIDTH - HEALTH_BAR_X_1 - HEALTH_BAR_WIDTH, 15
HEALTH_BAR_THICKNESS = 2




# Config for hp
FULL_HP = 100




def load_known_face(path):
    image = face_recognition.load_image_file(path)
    # image_encoding = face_recognition.face_encodings(image)[0]
    image_encoding = face_recognition.face_encodings(image)
    image_name = str(path).split('/')[-1].split('.')[0]
    return image_encoding[0], image_name

def load_known_faces(opt):
    known_face_encodings = []
    known_face_names = []
    known_player = {}

    files = [file for file in os.listdir(opt.known_face_dir) if os.path.isfile(os.path.join(opt.known_face_dir, file))]
    print(files)
    for file in files:
        image_encoding, image_name = load_known_face(os.path.join(opt.known_face_dir, file))
        known_face_encodings.append(image_encoding)
        known_face_names.append(image_name)
        known_player[image_name] = Player(image_name, player=1, opt=opt)
    return known_face_encodings, known_face_names, len(known_face_names), known_player


def face_detection(frame, known_face_encodings, known_face_names, known_face_num, known_player):
    # Resize frame of video to 1/4 size for faster face recognition processing
    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_small_frame = frame
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
    
    if known_face_num > 0:
        sorted_face_names = sorted(known_face_names)
        player_1 = known_player[sorted_face_names[0]]
        player_1.display_face()
        player_1.draw_blood_bar()
        if len(face_names) == 0:
            pass
        elif face_names[0] == 'Unknown':
            pass
        else:
            player_2 = known_player[face_names[0]]
            player_2.display_face()
            player_2.draw_blood_bar()

    for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4 * WINDOW_HEIGHT / ORIG_WINDOW_HEIGHT
            right *= 4 * WINDOW_HEIGHT / ORIG_WINDOW_HEIGHT
            bottom *= 4 * WINDOW_HEIGHT / ORIG_WINDOW_HEIGHT
            left *= 4 * WINDOW_HEIGHT / ORIG_WINDOW_HEIGHT

            
            demography = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            emotion_info = demography[0]['emotion']
            emo = max(emotion_info, key=emotion_info.get)


            if name == 'Unknown':
                pygame.draw.rect(screen, RED, (left, top, (right - left), (bottom - top)))
                # # Draw a box around the face
                # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # # Draw a label with a name below the face
                # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                # font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(frame, name + '\n' + emo, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            else:
                pygame.draw.rect(screen, GREEN, (left, top, (right - left), (bottom - top)))
                # # Draw a box around the face
                # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # # Draw a label with a name below the face
                # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                # font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    if len(face_names) == 0:
        return 'None', 'None', face_locations, face_names
    elif face_names[0]:
        return face_names[0], emo, face_locations, face_names
    

class Player:
    def __init__(self, name, player, opt):
        self.name = name
        self.blood = FULL_HP
        self.target_face = None
        self.player = player
        self.opt = opt
        self.changed = False
        self.card_point = []

        target_face_names = [str(file).split('.')[0] for file in os.listdir(opt.target_face_dir) if file.endswith('.jpg')]
        self.target_face = random.choice(target_face_names)

    def draw_blood_bar(self):
        ratio = self.blood / FULL_HP
        if self.player == 1:
            pygame.draw.rect(screen, WHITE, (HEALTH_BAR_X_1 - HEALTH_BAR_THICKNESS, HEALTH_BAR_Y_1 - HEALTH_BAR_THICKNESS, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT))
            pygame.draw.rect(screen, RED, (HEALTH_BAR_X_1, HEALTH_BAR_Y_1, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT))
            pygame.draw.rect(screen, GREEN, (HEALTH_BAR_X_1, HEALTH_BAR_Y_1, HEALTH_BAR_WIDTH * ratio, HEALTH_BAR_HEIGHT))    
        elif self.player == 2:
            pygame.draw.rect(screen, WHITE, (HEALTH_BAR_X_2 - HEALTH_BAR_THICKNESS, HEALTH_BAR_Y_2 - HEALTH_BAR_THICKNESS, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT))
            pygame.draw.rect(screen, RED, (HEALTH_BAR_X_2, HEALTH_BAR_Y_2, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT))
            pygame.draw.rect(screen, GREEN, (HEALTH_BAR_X_2 + HEALTH_BAR_WIDTH * (1 - ratio), HEALTH_BAR_Y_2, HEALTH_BAR_WIDTH * ratio, HEALTH_BAR_HEIGHT))
        else:
            raise Exception('Wrong player index!!')
        
    def display_face(self):    
        face_path = os.path.join(opt.changed_face_dir, self.name + '.jpg') if self.changed else os.path.join(opt.known_face_dir, self.name + '.jpg')
        image = pygame.image.load(face_path)
        image = pygame.transform.scale(image, (DISPLAYED_IMG_SIZE, DISPLAYED_IMG_SIZE))
        if self.player == 1:
            screen.blit(image, (DISPLAYED_IMG_X_1, DISPLAYED_IMG_Y_1))
        elif self.player == 2:
            screen.blit(image, (DISPLAYED_IMG_X_2, DISPLAYED_IMG_Y_2))
        else:
            raise Exception('Wrong player index!!')
    def draw_text(self, text, font, text_col, x, y):
        img = font.render(text, True, text_col)
        screen.blit(img, (x, y))
    def update(self):
        if not self.changed:
            self.changed = not self.changed
        self.blood = random.randint(0, self.blood)

def add_faces(known_face_encodings, known_face_names, known_face_num, known_player, face_locations, face_names, frame, opt):
    frame = np.transpose(frame, (1, 0, 2))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./test.jpg', frame)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        print(known_face_num)
        print(face_names)
        # if name != 'Unknown':
        #     continue
        top *= int(4 * WINDOW_HEIGHT / ORIG_WINDOW_HEIGHT)
        right *= int(4 * WINDOW_HEIGHT / ORIG_WINDOW_HEIGHT)
        bottom *= int(4 * WINDOW_HEIGHT / ORIG_WINDOW_HEIGHT)
        left *= int(4 * WINDOW_HEIGHT / ORIG_WINDOW_HEIGHT)


        print(top, left, bottom, right)
        print(frame.shape)
        face_image_new_raw = frame[top:bottom, left:right]
        face_image_new_raw = cv2.resize(face_image_new_raw, (opt.face_img_size, opt.face_img_size))
        print(face_image_new_raw.shape)
        # face_image_new_raw = cv2.resize(face_image_new_raw, (opt.face_img_size, opt.face_img_size))
        cv2.imwrite(os.path.join(opt.known_face_dir, f'{known_face_num}.jpg'), face_image_new_raw)
        face_image_new_encoding, face_image_new_name = load_known_face(os.path.join(opt.known_face_dir, f'{known_face_num}.jpg'))
        known_face_encodings.append(face_image_new_encoding)
        known_face_names.append(face_image_new_name)            
        if known_face_num > 0:
            known_player[str(known_face_num)] = Player(face_image_new_name, player=2, opt=opt)
        else:
            known_player[str(known_face_num)] = Player(face_image_new_name, player=1, opt=opt)
        known_face_num += 1

    return known_face_encodings, known_face_names, known_face_num, known_player

def main(opt):
    # Load known face

    if opt.known_face_dir is not None:
        os.makedirs(opt.known_face_dir, exist_ok=True)
        known_face_encodings, known_face_names, known_face_num, known_player = load_known_faces(opt)
        sorted_face_names = sorted(known_face_names)
        cprint.info(known_face_names)

    
    



    # Initialize variable of face
    face_locations = []
    face_encodings = []
    face_names = []

    past_timesteps_length = 10
    past_timesteps_threshold = 0.7
    past_timesteps = deque(maxlen=past_timesteps_length)

    game_mode = False

    player_1 = None
    player_2 = None

    while True:
    
        # print(game_mode)

        # Quit operation
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_q:
                    pygame.quit()
                    exit()
                elif event.key == K_1 and game_mode:
                    player_1.update()
                    change_face(player_1, 1 - player_1.blood / FULL_HP)
                    # change_face(player_1.name, player_1.target_face, player_1.opt, player_1.blood / FULL_HP)
                elif event.key == K_2 and game_mode:
                    player_2.update()
                    change_face(player_2, 1 - player_2.blood / FULL_HP)
                    
                    # change_face(player_2.name, player_2.target_face, player_2.opt, player_2.blood / FULL_HP)

        # Read a frame from the video capture
        ret, frame = video_capture.read()

        # Convert the OpenCV frame to Pygame surface
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (WINDOW_WIDTH, WINDOW_HEIGHT))
        small_frame_rgb = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2)) # Transpose image to fit form of pygame
        pygame_frame = pygame.surfarray.make_surface(frame_rgb)

        pygame_frame = pygame.transform.scale(pygame_frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        # Display the frame in the Pygame window
        screen.blit(pygame_frame, (0, 0))

        if not game_mode:
            
            opponet_name, opponent_emo, face_locations, face_names = face_detection(small_frame_rgb, known_face_encodings, known_face_names, known_face_num, known_player)
            past_timesteps.append({"name": opponet_name, "emo": opponent_emo})
            flattened_timesteps = [(item["name"], item["emo"]) for item in past_timesteps]
            timestep_counter = Counter(flattened_timesteps)
            most_common_combination, most_common_count = timestep_counter.most_common(1)[0]

            if most_common_count > past_timesteps_length * past_timesteps_threshold:
                if most_common_combination == ('Unknown', 'happy'):
                    known_face_encodings, known_face_names, known_face_num, known_player = add_faces(known_face_encodings, known_face_names, known_face_num, known_player, face_locations, face_names, frame_rgb, opt)
                    # Clear record of emotion in the past timestep, avoid record duplicate face
                    most_common_combination = None
                    past_timesteps.clear()
                    time.sleep(2)
                elif most_common_combination[0] != 'None' and most_common_combination[1] == 'happy':
                    # Start to play card game
                    player_1 = known_player[sorted_face_names[0]] # Player_1 
                    player_2 = known_player[most_common_combination[0]]
                    game_mode = not game_mode
                else:
                    pass
        else:
            # Poker card game part
            # display information about two player
            player_1.display_face()
            player_1.draw_blood_bar()
            player_2.display_face()
            player_2.draw_blood_bar()
            

            
        

        pygame.display.flip()

        # Control the frame rate
        clock.tick(fps)

    # Release the video capture and close Pygame
    video_capture.release()
    pygame.quit()


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
    opt = opt_parser()
    main(opt)