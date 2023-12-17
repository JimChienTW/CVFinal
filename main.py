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
from ultralytics import YOLO

# Initialize Pygame
pygame.init()

# Set up the Pygame window
ORIG_WINDOW_WIDTH, ORIG_WINDOW_HEIGHT = 1920, 1080
WINDOW_WIDTH, WINDOW_HEIGHT = 1440, 810
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Video Stream Player")

# Set up the Pygame text font object
font_size = 24
font = pygame.font.Font(None, font_size)  # You can replace 'None' with a font file path if you have a specific font

# # load music and sounds
# try:
#     pygame.mixer.music.load("assets/audio/roa-music-innocence.mp3")
#     pygame.mixer.music.set_volume(0.5)
#     pygame.mixer.music.play(-1, 0.0, 5000)
# except:
#     pass

# poker card number mapping dictionary
poker_card = ['Ah'
              , 'Kh', 'Qh', 'Jh', '10h', '9h', '8h', '7h', '6h', '5h', '4h', '3h', '2h', 'Ad'
              , 'Kd', 'Qd', 'Jd', '10d', '9d', '8d', '7d', '6d', '5d', '4d', '3d', '2d', 'Ac'
              , 'Kc', 'Qc', 'Jc', '10c', '9c', '8c', '7c', '6c', '5c', '4c', '3c', '2c', 'As'
              , 'Ks', 'Qs', 'Js', '10s', '9s', '8s', '7s', '6s', '5s', '4s', '3s', '2s']

def label_to_card_number(label_item):
    card_values = {
        'A': 10, 'K': 10, 'Q': 10, 'J': 10,
        '10': 10, '9': 9, '8': 8, '7': 7, '6': 6,
        '5': 5, '4': 4, '3': 3, '2': 2
    }
    label_item_int = int(label_item)
    # Ensure label_item is within the valid range
    if 0 <= label_item_int < len(poker_card):
        # Get the corresponding card from poker_card list
        card = poker_card[label_item_int]
        # Extract the rank from the card
        rank = card[:-1]  # Exclude the last character (suit)
        # Map the rank to its numerical value using the dictionary
        card_number = card_values.get(rank, 0)

        return card_number
    else:
        # Handle the case where label_item is out of range
        return 0

def check_game_result(hand1, hand2):
    # Convert each label item to its corresponding card number
    # Then sum up card numbers
    hand_values1 = sum([label_to_card_number(item) for item in hand1])
    hand_values2 = sum([label_to_card_number(item) for item in hand2])

    # player2 is the dealer, if player1 greater than 21, then player1 loses anyway.
    if hand_values1 > 21:
        loser = 1
    elif hand_values2 > 21:
        loser = 2
    elif hand_values1 == hand_values2:
        loser = 0
    elif hand_values1 < hand_values2:
        loser = 1
    elif hand_values2 < hand_values1:
        loser = 2
    
    return loser


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
ORANGE = (255, 165, 0)

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

# YOLO model
model = YOLO('./YOLO/best_n.pt')

def load_known_face(path):
    image = face_recognition.load_image_file(path)
    # image_encoding = face_recognition.face_encodings(image)[0]
    image_encoding = face_recognition.face_encodings(image)
    image_name = str(path).split('/')[-1].split('.')[0]
    encoding = image_encoding[0] if len(image_encoding) > 0 else None
    return encoding, image_name

def load_known_faces(opt):
    known_face_encodings = []
    known_face_names = []
    known_player = {}

    files = [file for file in os.listdir(opt.known_face_dir) if os.path.isfile(os.path.join(opt.known_face_dir, file))]
    print(files)
    for file in files:
        image_encoding, image_name = load_known_face(os.path.join(opt.known_face_dir, file))
        if image_encoding is None:
            continue
        known_face_encodings.append(image_encoding)
        known_face_names.append(image_name)
        known_player[image_name] = Player(image_name, player=1, opt=opt)
    return known_face_encodings, known_face_names, len(known_face_names), known_player

def draw_text(text, color, left, top):
    text = font.render(text, True, color)
    text_rect = text.get_rect(left=left, top=top)  
    screen.blit(text, text_rect)

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
    
    # Display blood and face of player
    if known_face_num > 0:
        sorted_face_names = sorted(known_face_names)
        player_1 = known_player[sorted_face_names[0]]
        player_1.display_face()
        player_1.draw_blood_bar(game_mode=False)
        if len(face_names) == 0:
            pass
        elif face_names[0] == 'Unknown':
            pass
        else:
            player_2 = known_player[face_names[0]]
            player_2.display_face()
            player_2.draw_blood_bar(game_mode=False)

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
            pygame.draw.rect(screen, RED, (left, top, (right - left), (bottom - top)), 3)
            draw_text((str(name) + ", " + str(emo)), RED, left, bottom + 10)
            # draw_text(name, RED, face_locations)
        #     # # Draw a box around the face
        #     # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #     # # Draw a label with a name below the face
        #     # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #     # font = cv2.FONT_HERSHEY_DUPLEX
        #     # cv2.putText(frame, name + '\n' + emo, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
            pygame.draw.rect(screen, GREEN, (left, top, (right - left), (bottom - top)), 3)
            draw_text((str(name) + ", " + str(emo)), GREEN, left, bottom + 10)
            # draw_text(name, GREEN, face_locations)

        #     # # Draw a box around the face
        #     # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        #     # # Draw a label with a name below the face
        #     # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        #     # font = cv2.FONT_HERSHEY_DUPLEX
        #     # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    if len(face_names) == 0:
        return 'None', 'None', face_locations, face_names
    elif face_names[0]:
        return face_names[0], emo, face_locations, face_names
    
class Player:
    def __init__(self, name, player, opt):
        self.opt = opt
        self.name = name
        self.blood = FULL_HP
        self.game_round = opt.game_round
        self.heart = opt.game_round
        self.target_face = None
        self.player = player
        self.changed = False
        self.card_point = []

        target_face_names = [str(file).split('.')[0] for file in os.listdir(opt.target_face_dir) if file.endswith('.jpg')]
        self.target_face = random.choice(target_face_names)

    def draw_blood_bar(self, game_mode = False):
        ratio = self.blood / FULL_HP
        # draw blood bar for player1
        if self.player == 1:
            pygame.draw.rect(screen, WHITE, (HEALTH_BAR_X_1 - HEALTH_BAR_THICKNESS, HEALTH_BAR_Y_1 - HEALTH_BAR_THICKNESS, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT))
            pygame.draw.rect(screen, RED, (HEALTH_BAR_X_1, HEALTH_BAR_Y_1, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT))
            pygame.draw.rect(screen, GREEN, (HEALTH_BAR_X_1, HEALTH_BAR_Y_1, HEALTH_BAR_WIDTH * ratio, HEALTH_BAR_HEIGHT))    
        # draw blood bar for player2,
        elif self.player == 2:
            # game_mode: red ; not game_mode: orange
            if game_mode:
                pygame.draw.rect(screen, WHITE, (HEALTH_BAR_X_2 - HEALTH_BAR_THICKNESS, HEALTH_BAR_Y_2 - HEALTH_BAR_THICKNESS, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT))
                pygame.draw.rect(screen, RED, (HEALTH_BAR_X_2, HEALTH_BAR_Y_2, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT))
                pygame.draw.rect(screen, GREEN, (HEALTH_BAR_X_2 + HEALTH_BAR_WIDTH * (1 - ratio), HEALTH_BAR_Y_2, HEALTH_BAR_WIDTH * ratio, HEALTH_BAR_HEIGHT))
            elif not game_mode:
                pygame.draw.rect(screen, WHITE, (HEALTH_BAR_X_2 - HEALTH_BAR_THICKNESS, HEALTH_BAR_Y_2 - HEALTH_BAR_THICKNESS, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT))
                pygame.draw.rect(screen, RED, (HEALTH_BAR_X_2, HEALTH_BAR_Y_2, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT))
                pygame.draw.rect(screen, ORANGE, (HEALTH_BAR_X_2 + HEALTH_BAR_WIDTH * (1 - ratio), HEALTH_BAR_Y_2, HEALTH_BAR_WIDTH * ratio, HEALTH_BAR_HEIGHT))
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
    
    def update(self):
        if not self.changed:
            self.changed = not self.changed
        if self.heart > 0 :
            # load sword sound and play
            sword = pygame.mixer.Sound("assets/audio/sword.wav")
            sword.set_volume(1)
            sword.play()

            self.heart -= 1
            self.blood = FULL_HP * (self.heart / self.game_round)
        else:
            print("you're out, stop playing game!!")
    
    def die(self):
        # load game over sound and play
        game_over = pygame.mixer.Sound("assets/audio/game_over.wav")
        game_over.set_volume(1) 
        duration = (game_over.get_length()-3) * 1000  # Convert duration to milliseconds
        game_over.play()
        pygame.time.wait(int(duration))

        # load vicory image and display, wait for 5 sec
        lose_image = pygame.image.load(os.path.join(opt.changed_face_dir, self.name + '.jpg')).convert_alpha()
        lose_image = pygame.transform.scale(lose_image, (WINDOW_WIDTH, WINDOW_HEIGHT))  # Resize image to screen dimensions
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Game Over")
        screen.blit(lose_image, (0, 0))
        pygame.display.flip()  # Update the display
        pygame.time.delay(5000)

def add_faces(known_face_encodings, known_face_names, known_face_num, known_player, face_locations, face_names, frame, opt):
    frame = np.transpose(frame, (1, 0, 2))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./test.jpg', frame)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # print(known_face_num)
        # print(face_names)
        # if name != 'Unknown':
        #     continue
        top *= int(4 * WINDOW_HEIGHT / ORIG_WINDOW_HEIGHT)
        right *= int(4 * WINDOW_HEIGHT / ORIG_WINDOW_HEIGHT)
        bottom *= int(4 * WINDOW_HEIGHT / ORIG_WINDOW_HEIGHT)
        left *= int(4 * WINDOW_HEIGHT / ORIG_WINDOW_HEIGHT)

        # print(top, left, bottom, right)
        # print(frame.shape)
        face_image_new_raw = frame[top:bottom, left:right]
        face_image_new_raw = cv2.resize(face_image_new_raw, (opt.face_img_size, opt.face_img_size))
        # print(face_image_new_raw.shape)
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

    return known_face_encodings, known_face_names, known_face_num, known_player, face_image_new_name

def main(opt):

    # Remove all known_face
    os.system(f'rm -rf {opt.known_face_dir}')    

    debug_list = []

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
    past_timesteps_threshold = 0.5
    past_timesteps = deque(maxlen=past_timesteps_length)

    game_mode = False
    end_game = False

    player_1 = None
    player_2 = None

    most_common_combination = None

    while True:
    
        print(debug_list)
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
                elif event.key == K_1:
                    player_1.update()
                    change_face(player_1, 1 - player_1.blood / FULL_HP)
                    # change_face(player_1.name, player_1.target_face, player_1.opt, player_1.blood / FULL_HP)
                elif event.key == K_2 :
                    player_2.update()
                    change_face(player_2, 1 - player_2.blood / FULL_HP)
                elif event.key == K_3:
                    # game_over.play()                 
                    player_1.update()
                    player_2.update()
                    change_face(player_2, 1 - player_2.blood / FULL_HP)

        # Read a frame from the video capture
        _, frame = video_capture.read()

        # Convert the OpenCV frame to Pygame surface
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (WINDOW_WIDTH, WINDOW_HEIGHT)) # resize image fit with window size
        small_frame_rgb = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # downsizing image as input image for face detection
        frame_rgb_transposed = np.transpose(frame_rgb, (1, 0, 2)) # Transpose image to fit form of pygame. Flip the image
        pygame_frame = pygame.surfarray.make_surface(frame_rgb_transposed)

        pygame_frame = pygame.transform.scale(pygame_frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        # Display the background frame in the Pygame window
        screen.blit(pygame_frame, (0, 0))

        print(most_common_combination)

        opponet_name, opponent_emo, face_locations, face_names = face_detection(small_frame_rgb, known_face_encodings, known_face_names, known_face_num, known_player)

        if not game_mode:
            past_timesteps.append({"name": opponet_name, "emo": opponent_emo})
            flattened_timesteps = [(item["name"], item["emo"]) for item in past_timesteps]
            timestep_counter = Counter(flattened_timesteps)
            most_common_combination, most_common_count = timestep_counter.most_common(1)[0]

            if most_common_count > past_timesteps_length * past_timesteps_threshold:
                if most_common_combination[1] == 'happy':
                    if most_common_combination[0] == 'Unknown':
                        known_face_encodings, known_face_names, known_face_num, known_player, new_face_name = add_faces(known_face_encodings, known_face_names, known_face_num, known_player, face_locations, face_names, frame_rgb_transposed, opt)
                        # Clear record of emotion in the past timestep, avoid record duplicate face
                        most_common_combination = None
                        past_timesteps.clear()
                        time.sleep(2)
                    # Start to play card game
                    if known_face_num > 1:
                        sorted_face_names = sorted(known_face_names)
                        player_1 = known_player[sorted_face_names[0]] # Player_1 
                        player_2 = known_player[most_common_combination[0]] if most_common_combination != None else known_player[new_face_name]
                        
                        past_timesteps.clear()
                        if player_2.blood > 0:
                            game_mode = not game_mode

                        # print("game mode change by start playing game", game_mode)
                        debug_list.append("game mode change by start playing game " + str(game_mode))
                else:
                    pass

        else:
            # Poker card game part
            # display information about two player
            player_1.display_face()
            player_1.draw_blood_bar(game_mode=True)
            player_2.display_face()
            player_2.draw_blood_bar(game_mode=True)
            frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            results = model(frame_resized, show = False)
            
            card_1 = set()
            card_num_1 = []
            card_2 = set()
            card_num_2 = []

            for box, label in zip(results[0].boxes.xyxyn.cpu().numpy(), results[0].boxes.cls.cpu().numpy()):
                # print(box)
                if ((box[1] + box[3]) / 2) < 1 and ((box[1] + box[3]) / 2) > 0.5:
                    card_1.add(label.item())
                    card_num_1 = []
                    card_num_1 = list(label_to_card_number(item) for item in card_1)
                    card_number_str_1 = ' '.join(map(str,card_num_1))
                    
                    draw_text(card_number_str_1, RED, HEALTH_BAR_X_1, HEALTH_BAR_Y_1 + HEALTH_BAR_HEIGHT + 10)
                elif ((box[1] + box[3]) / 2) < 0.5:
                    card_2.add(label.item())
                    card_num_2 = []
                    card_num_2 = list(label_to_card_number(item) for item in card_2)
                    card_number_str_2 = ' '.join(map(str,card_num_2))
 
                    draw_text(card_number_str_2, RED, HEALTH_BAR_X_2, HEALTH_BAR_Y_2 + HEALTH_BAR_HEIGHT + 10)
                else:
                    pass

            # card game end, decide game winner
            card_detect_threshold = 0.5
            card_detect_timesteps = deque(maxlen=10)
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    # finish one round
                    if event.key == K_SPACE:
                        result =  check_game_result(card_1, card_2)
                        card_detect_timesteps.append({"result": result})
                        flattened_card_timesteps = [(item["result"]) for item in card_detect_timesteps]
                        card_timestep_counter = Counter(flattened_card_timesteps)
                        stable_result , most_common_count = card_timestep_counter.most_common(1)[0]
                        if most_common_count > len(card_detect_timesteps) * card_detect_threshold:
                            # game end in a draw
                            if not stable_result:
                                game_mode = not game_mode
                                debug_list.append("game mode change by not result " + str(game_mode))
                            # player1 lose the game
                            elif stable_result == 1:
                                print("player_1 lose the game")
                                player_1.update()
                                change_face(player_1, (1 - player_1.blood / FULL_HP) * 0.5)
                                card_1.clear()
                                card_2.clear() 
                                # player1 died
                                if player_1.blood == 0:
                                    end_game = True
                                else:
                                    game_mode = not game_mode
                                debug_list.append("game mode change by result1 " + str(game_mode))
                            # player2 lose the game
                            elif stable_result == 2:
                                print("player_2 lose the game")
                                player_2.update()
                                change_face(player_2, (1 - player_2.blood / FULL_HP) * 0.5)
                                card_1.clear()
                                card_2.clear()
                                # player2 died
                                if player_2.blood == 0:
                                    player_2.die()
                                game_mode = not game_mode
                                debug_list.append("game mode change by result2 " + str(game_mode))

            if end_game:
                player_1.die()
                video_capture.release()
                pygame.quit()
                exit()
       
            # print(card_1)
            print(card_num_1)
            # print(card_2)
            print(card_num_2)
            
        pygame.display.flip()

        # Control the frame rate
        clock.tick(fps)
    
    # Release the video capture and close Pygame
    video_capture.release()
    pygame.quit()

def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--known_face_dir', type=str, default='face_base')
    parser.add_argument('--target_face_dir', type=str, default='assets/target_face')
    parser.add_argument('--changed_face_dir', type=str, default='changed_face')
    parser.add_argument('--keypoints_dir', type=str, default='keypoints')
    parser.add_argument('--tri_dir', type=str, default='tri')
    parser.add_argument('--face_img_size', type=int, default=640)
    parser.add_argument('--game_round', type=int, default=4)
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = opt_parser()
    main(opt)