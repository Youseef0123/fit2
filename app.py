from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
import time
app = Flask(__name__)

# Initialize pygame mixer
pygame.mixer.init()
# Load sound
sound = pygame.mixer.Sound(r"D:\siren-alert-96052.mp3")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    ab = a - b
    bc = c - b

    # Calculate the angle
    angle = np.arccos(np.clip(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)), -1.0, 1.0))
    return math.degrees(angle)





def hummer():
    left_counter = 0  # Counter for left arm
    right_counter = 0  # Counter for right arm
    left_state = None  # State for left arm
    right_state = None  # State for right arm
    cap = cv2.VideoCapture(0)
    sound_playing = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Define landmarks for both arms
            arm_sides = {
                'left': {
                    'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                    'hip': mp_pose.PoseLandmark.LEFT_HIP
                },
                'right': {
                    'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                    'hip': mp_pose.PoseLandmark.RIGHT_HIP
                }
            }

            # Get coordinates for both shoulders and both hips
            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]
            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]
            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]

            # Draw a line between both shoulders
            cv2.line(
                image,
                (int(left_shoulder[0] * image.shape[1]), int(left_shoulder[1] * image.shape[0])),
                (int(right_shoulder[0] * image.shape[1]), int(right_shoulder[1] * image.shape[0])),
                (0, 255, 255),  # Color: yellow
                2
            )

            # Draw a line between both hips
            cv2.line(
                image,
                (int(left_hip[0] * image.shape[1]), int(left_hip[1] * image.shape[0])),
                (int(right_hip[0] * image.shape[1]), int(right_hip[1] * image.shape[0])),
                (0, 255, 255),  # Color: yellow
                2
            )

            # Initialize flags for both arms' angle violations
            arm_violated = {'left': False, 'right': False}

            for side, joints in arm_sides.items():
                # Get coordinates for each side
                shoulder = [
                    landmarks[joints['shoulder'].value].x,
                    landmarks[joints['shoulder'].value].y,
                ]
                elbow = [
                    landmarks[joints['elbow'].value].x,
                    landmarks[joints['elbow'].value].y,
                ]
                wrist = [
                    landmarks[joints['wrist'].value].x,
                    landmarks[joints['wrist'].value].y,
                ]
                hip = [
                    landmarks[joints['hip'].value].x,
                    landmarks[joints['hip'].value].y
                ]

                # Calculate angles
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_angle = calculate_angle(hip, shoulder, elbow)

                # Draw connections for the arm
                arm_connections = [
                    (joints['shoulder'], joints['elbow']),
                    (joints['elbow'], joints['wrist'])
                ]
                torso_connections = [
                    (joints['hip'], joints['shoulder'])
                ]

                joint_positions = {
                    'Shoulder': [shoulder[0] * image.shape[1], shoulder[1] * image.shape[0]],
                    'Elbow': [elbow[0] * image.shape[1], elbow[1] * image.shape[0]],
                    'Wrist': [wrist[0] * image.shape[1], wrist[1] * image.shape[0]],
                    'Hip': [hip[0] * image.shape[1], hip[1] * image.shape[0]]
                }

                # Draw arm connections
                for connection in arm_connections:
                    start_idx = connection[0].value
                    end_idx = connection[1].value

                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]

                    start_coords = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
                    end_coords = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))

                    cv2.line(image, start_coords, end_coords,  (0,255,0), 2)

                # Draw torso connections
                for connection in torso_connections:
                    start_idx = connection[0].value
                    end_idx = connection[1].value

                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]

                    start_coords = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
                    end_coords = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))


                    cv2.line(image, start_coords, end_coords, (0,255,0), 2)  # Different color for torso

                # Draw joints
                for joint, position in joint_positions.items():
                    cv2.circle(image, (int(position[0]), int(position[1])), 7, (0, 0, 255), -1)

                # Display angles
                cv2.putText(
                    image,
                    f' {int(elbow_angle)}',
                    tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    image,
                    f' {int(shoulder_angle)}',
                    tuple(np.multiply(shoulder, [image.shape[1], image.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # Check if the angles are outside the desired range
                elbow_max = 180
                shoulder_max = 30
                sagittal_angle_threshold = 90
                shoulder_max_back = 25  # Maximum angle for shoulder extension backward
                elbow_min_back = 0  

                if elbow_angle > elbow_max or shoulder_angle >= shoulder_max:
                    arm_violated[side] = True
                if elbow_angle < elbow_min_back or shoulder_angle > shoulder_max_back:
                     arm_violated[side] = True

                # if elbow_angle < sagittal_angle_threshold or shoulder_angle > sagittal_angle_threshold:
                if shoulder_angle> sagittal_angle_threshold:
                    arm_violated[side] = True
                if not arm_violated['left'] and not arm_violated['right']:
                    if side == 'left':
                        if elbow_angle > 160:
                            left_state = 'down'
                        if elbow_angle < 30 and left_state == 'down':
                            left_state = 'up'
                            left_counter += 1
                            print(f'Left Counter: {left_counter}')
                    if side == 'right':
                        if elbow_angle > 160:
                            right_state = 'down'
                        if elbow_angle < 30 and right_state == 'down':
                            right_state = 'up'
                            right_counter += 1
                            print(f'Right Counter: {right_counter}')

            # Play sound if either arm is violated and not already playing
            if any(arm_violated.values()) and not sound_playing:
                sound.play()
                sound_playing = True
            elif not any(arm_violated.values()) and sound_playing:
                sound.stop()
                sound_playing = False

            # Draw counters on the image
            cv2.putText(image, f'Left Counter: {left_counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Right Counter: {right_counter}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Convert the image to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        # Yield the frame to the Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        


#the front Raise for shoulder  /////////////////////////////////////////////////////
def dumbbell_front_raise():
    left_counter = 0
    right_counter = 0
    left_state = "down"
    right_state = "down"
    cap = cv2.VideoCapture(0)
    sound_playing = False  # Add flag to track if sound is playing

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            arm_sides = {
                'left': {
                    'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                    'hip': mp_pose.PoseLandmark.LEFT_HIP
                },
                'right': {
                    'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                    'hip': mp_pose.PoseLandmark.RIGHT_HIP
                }
            }

            arm_angle_violated = False

            for side, joints in arm_sides.items():
                shoulder = landmarks[joints['shoulder'].value]
                elbow = landmarks[joints['elbow'].value]
                wrist = landmarks[joints['wrist'].value]
                hip = landmarks[joints['hip'].value]

                shoulder_coords = (int(shoulder.x * image.shape[1]), int(shoulder.y * image.shape[0]))
                elbow_coords = (int(elbow.x * image.shape[1]), int(elbow.y * image.shape[0]))
                wrist_coords = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
                hip_coords = (int(hip.x * image.shape[1]), int(hip.y * image.shape[0]))

                cv2.line(image, shoulder_coords, elbow_coords, (0, 255, 0), 2)
                cv2.line(image, elbow_coords, wrist_coords, (0, 255, 0), 2)
                cv2.line(image, hip_coords, shoulder_coords, (0, 255, 0), 2)

                for point in [shoulder_coords, elbow_coords, wrist_coords, hip_coords]:
                    cv2.circle(image, point, 7, (0, 0, 255), -1)

                elbow_angle = calculate_angle([shoulder.x, shoulder.y], [elbow.x, elbow.y], [wrist.x, wrist.y])
                shoulder_angle = calculate_angle([hip.x, hip.y], [shoulder.x, shoulder.y], [elbow.x, elbow.y])

                cv2.putText(image, f'{int(elbow_angle)}', elbow_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, f'{int(shoulder_angle)}', shoulder_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Check if shoulder angle exceeds 160 degrees and mark as violated
                if shoulder_angle > 150:
                    arm_angle_violated = True
                    # Highlight the angle in red to indicate violation
                    cv2.putText(image, f'{int(shoulder_angle)}', shoulder_coords, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                wrist_x = wrist.x * image.shape[1]
                shoulder_x = shoulder.x * image.shape[1]
                wrist_y = wrist.y * image.shape[0]
                shoulder_y = shoulder.y * image.shape[0]

                # ==== تعديل حساب العدّة ====
                if side == 'left':
                    if elbow_angle >= 110 and left_state == "down":
                        if wrist.y < shoulder.y and 30 < abs(wrist_x - shoulder_x) < 100:
                            left_state = "up"
                            left_counter += 1
                    elif elbow_angle > 160 and wrist.y > shoulder.y and left_state == "up":
                        left_state = "down"

                elif side == 'right':
                    if elbow_angle >= 110 and right_state == "down":
                        if wrist.y < shoulder.y and 30 < abs(wrist_x - shoulder_x) < 100:
                            right_state = "up"
                            right_counter += 1
                    elif elbow_angle > 160 and wrist.y > shoulder.y and right_state == "up":
                        right_state = "down"

            # Control the sound alert
            if arm_angle_violated and not sound_playing:
                sound.play()
                sound_playing = True
            elif not arm_angle_violated and sound_playing:
                sound.stop()
                sound_playing = False
            
            # Display instruction message whenever the angle is violated (while alert is active)
            if sound_playing:
                # Add instruction message to lower arms when angle is too high
                # Centered text with background for better visibility
                text = "LOWER YOUR ARM!"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = image.shape[0] // 2
                
                # Draw semi-transparent background for text
                overlay = image.copy()
                cv2.rectangle(overlay, 
                             (text_x - 10, text_y - text_size[1] - 10),
                             (text_x + text_size[0] + 10, text_y + 10),
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                
                # Draw text
                cv2.putText(image, text, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(image, f'Left Counter: {left_counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Right Counter: {right_counter}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.waitKey(1)
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



#the Squat for leg //////////////////////////////////////////////////////////
# def squat():
#     counter = 0  # Counter for squats
#     state = None  # State for squat position
#     cap = cv2.VideoCapture(0)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Flip the frame horizontally
#         frame = cv2.flip(frame, 1)
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(image)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark

#             # Define landmarks for both legs
#             leg_sides = {
#                 'left': {
#                     'hip': mp_pose.PoseLandmark.LEFT_HIP,
#                     'knee': mp_pose.PoseLandmark.LEFT_KNEE,
#                     'ankle': mp_pose.PoseLandmark.LEFT_ANKLE
#                 },
#                 'right': {
#                     'hip': mp_pose.PoseLandmark.RIGHT_HIP,
#                     'knee': mp_pose.PoseLandmark.RIGHT_KNEE,
#                     'ankle': mp_pose.PoseLandmark.RIGHT_ANKLE
#                 }
#             }

#             for side, joints in leg_sides.items():
#                 # Get coordinates for each side
#                 hip = [
#                     landmarks[joints['hip'].value].x,
#                     landmarks[joints['hip'].value].y,
#                 ]
#                 knee = [
#                     landmarks[joints['knee'].value].x,
#                     landmarks[joints['knee'].value].y,
#                 ]
#                 ankle = [
#                     landmarks[joints['ankle'].value].x,
#                     landmarks[joints['ankle'].value].y,
#                 ]

#                 # Convert normalized coordinates to image coordinates
#                 hip_coords = (int(hip[0] * image.shape[1]), int(hip[1] * image.shape[0]))
#                 knee_coords = (int(knee[0] * image.shape[1]), int(knee[1] * image.shape[0]))
#                 ankle_coords = (int(ankle[0] * image.shape[1]), int(ankle[1] * image.shape[0]))

#                 # Draw lines between hip, knee, and ankle
#                 cv2.line(image, hip_coords, knee_coords, (0, 255, 0), 2)  # Green line
#                 cv2.line(image, knee_coords, ankle_coords, (0, 255, 0), 2)  # Green line

#                 # Draw circles at hip, knee, and ankle
#                 cv2.circle(image, hip_coords, 7, (0, 0, 255), -1)  # Red circle
#                 cv2.circle(image, knee_coords, 7, (0, 0, 255), -1)  # Red circle
#                 cv2.circle(image, ankle_coords, 7, (0, 0, 255), -1)  # Red circle

#                 # Calculate angles
#                 knee_angle = calculate_angle(hip, knee, ankle)

#                 # Display angles
#                 cv2.putText(
#                     image,
#                     f' {int(knee_angle)}',
#                     knee_coords,
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (255, 255, 255),
#                     2,
#                     cv2.LINE_AA
#                 )

#                 # Check if the knee angle is within the desired range for a squat
#                 if knee_angle < 90:
#                     state = "down"
#                 if knee_angle > 160 and state == "down":
#                     state = "up"
#                     counter += 1
#                     print(f'Squat Counter: {counter}')

#             # Draw counter on the image
#             cv2.putText(image, f'Squat Counter: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         # Convert the image to JPEG format for streaming
#         ret, buffer = cv2.imencode('.jpg', image)
#         frame = buffer.tobytes()

#         # Yield the frame to the Flask response
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def squat():
    counter = 0  # Counter for squats
    state = None  # State for squat position
    cap = cv2.VideoCapture(0)
    sound_playing = False  # Add flag to track sound state
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        angle_too_low = False  # Flag to track if angle is too low
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Define landmarks for both legs
            leg_sides = {
                'left': {
                    'hip': mp_pose.PoseLandmark.LEFT_HIP,
                    'knee': mp_pose.PoseLandmark.LEFT_KNEE,
                    'ankle': mp_pose.PoseLandmark.LEFT_ANKLE
                },
                'right': {
                    'hip': mp_pose.PoseLandmark.RIGHT_HIP,
                    'knee': mp_pose.PoseLandmark.RIGHT_KNEE,
                    'ankle': mp_pose.PoseLandmark.RIGHT_ANKLE
                }
            }
            
            for side, joints in leg_sides.items():
                # Get coordinates for each side
                hip = [
                    landmarks[joints['hip'].value].x,
                    landmarks[joints['hip'].value].y,
                ]
                knee = [
                    landmarks[joints['knee'].value].x,
                    landmarks[joints['knee'].value].y,
                ]
                ankle = [
                    landmarks[joints['ankle'].value].x,
                    landmarks[joints['ankle'].value].y,
                ]
                
                # Convert normalized coordinates to image coordinates
                hip_coords = (int(hip[0] * image.shape[1]), int(hip[1] * image.shape[0]))
                knee_coords = (int(knee[0] * image.shape[1]), int(knee[1] * image.shape[0]))
                ankle_coords = (int(ankle[0] * image.shape[1]), int(ankle[1] * image.shape[0]))
                
                # Draw lines between hip, knee, and ankle
                cv2.line(image, hip_coords, knee_coords, (0, 255, 0), 2)  # Green line
                cv2.line(image, knee_coords, ankle_coords, (0, 255, 0), 2)  # Green line
                
                # Draw circles at hip, knee, and ankle
                cv2.circle(image, hip_coords, 7, (0, 0, 255), -1)  # Red circle
                cv2.circle(image, knee_coords, 7, (0, 0, 255), -1)  # Red circle
                cv2.circle(image, ankle_coords, 7, (0, 0, 255), -1)  # Red circle
                
                # Calculate angles
                knee_angle = calculate_angle(hip, knee, ankle)
                
                # Display angles
                cv2.putText(
                    image,
                    f' {int(knee_angle)}',
                    knee_coords,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Check if the knee angle is less than 70 degrees (90-20)
                if knee_angle < 70:
                    angle_too_low = True
                
                # Check for squat logic
                if knee_angle < 90:
                    state = "down"
                if knee_angle > 160 and state == "down":
                    state = "up"
                    counter += 1
                    print(f'Squat Counter: {counter}')
            
            # Draw counter on the image
            cv2.putText(image, f'Squat Counter: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Handle alert for angle too low
            if angle_too_low:
                # Play alert sound if not already playing
                if not sound_playing:
                    sound.play()
                    sound_playing = True
                
                # Add message with better visibility
                warning_text = "WARNING! Knee angle too low. Adjust your position!"
                text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = image.shape[0] // 2
                
                # Add semi-transparent background for better text visibility
                overlay = image.copy()
                cv2.rectangle(overlay, 
                             (text_x - 10, text_y - text_size[1] - 10),
                             (text_x + text_size[0] + 10, text_y + 10),
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                
                # Draw warning text
                cv2.putText(image, warning_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                # Stop sound if it was playing
                if sound_playing:
                    sound.stop()
                    sound_playing = False
        
        # Convert the image to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        # Yield the frame to the Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



# shoulder_press       //////////////////////////////////////////////////
def shoulder_press():
    counter = 0  # عداد واحد لكلتا الذراعين
    stage = None  # حالة التمرين
    cap = cv2.VideoCapture(0)
    sound_playing = False
    
    # إضافة طباعة لتصحيح الأخطاء
    print("Shoulder Press Exercise Started")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        form_violated = False
        low_elbow_angle = False  # Flag for low elbow angle
        instruction_message = ""
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Define landmarks for both arms
            arm_sides = {
                'left': {
                    'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                    'hip': mp_pose.PoseLandmark.LEFT_HIP
                },
                'right': {
                    'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                    'hip': mp_pose.PoseLandmark.RIGHT_HIP
                }
            }
            
            # Variables to track whether both arms are in correct position
            left_arm_down = False
            right_arm_down = False
            left_arm_at_150 = False
            right_arm_at_150 = False
            
            left_elbow_angle = 0
            right_elbow_angle = 0
            
            # Process each arm
            for side, joints in arm_sides.items():
                # Get coordinates for each joint
                shoulder = [
                    landmarks[joints['shoulder'].value].x,
                    landmarks[joints['shoulder'].value].y
                ]
                elbow = [
                    landmarks[joints['elbow'].value].x,
                    landmarks[joints['elbow'].value].y
                ]
                wrist = [
                    landmarks[joints['wrist'].value].x,
                    landmarks[joints['wrist'].value].y
                ]
                hip = [
                    landmarks[joints['hip'].value].x,
                    landmarks[joints['hip'].value].y
                ]
                
                # Convert to pixel coordinates
                shoulder_coords = (int(shoulder[0] * image.shape[1]), int(shoulder[1] * image.shape[0]))
                elbow_coords = (int(elbow[0] * image.shape[1]), int(elbow[1] * image.shape[0]))
                wrist_coords = (int(wrist[0] * image.shape[1]), int(wrist[1] * image.shape[0]))
                hip_coords = (int(hip[0] * image.shape[1]), int(hip[1] * image.shape[0]))
                
                # Draw arm lines
                cv2.line(image, shoulder_coords, elbow_coords, (0, 255, 0), 2)
                cv2.line(image, elbow_coords, wrist_coords, (0, 255, 0), 2)
                cv2.line(image, shoulder_coords, hip_coords, (0, 255, 0), 2)
                
                # Draw joint circles
                cv2.circle(image, shoulder_coords, 7, (0, 0, 255), -1)
                cv2.circle(image, elbow_coords, 7, (0, 0, 255), -1)
                cv2.circle(image, wrist_coords, 7, (0, 0, 255), -1)
                cv2.circle(image, hip_coords, 7, (0, 0, 255), -1)
                
                # Calculate angles
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_angle = calculate_angle(hip, shoulder, elbow)
                
                # Store elbow angles for each arm
                if side == 'left':
                    left_elbow_angle = elbow_angle
                else:
                    right_elbow_angle = elbow_angle
                
                # Display angles
                elbow_color = (255, 255, 255)  # Default white
                
                # Check if elbow angle is too low (30 degrees or less)
                if elbow_angle <= 30:
                    low_elbow_angle = True
                    elbow_color = (0, 0, 255)  # Red when angle is too low
                    
                # Check if at target angle for UP position (around 150 degrees)
                elif 140 <= elbow_angle <= 160:
                    elbow_color = (0, 255, 0)  # Green when at target angle
                    if side == 'left':
                        left_arm_at_150 = True
                    else:
                        right_arm_at_150 = True
                
                # Check if at target angle for DOWN position (around 40 degrees)
                elif 35 <= elbow_angle <= 45:
                    elbow_color = (0, 255, 255)  # Yellow when at down position
                    if side == 'left':
                        left_arm_down = True
                    else:
                        right_arm_down = True
                
                cv2.putText(
                    image,
                    f'E: {int(elbow_angle)}°',
                    elbow_coords,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    elbow_color,
                    2,
                    cv2.LINE_AA
                )
                
                cv2.putText(
                    image,
                    f'S: {int(shoulder_angle)}°',
                    shoulder_coords,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Check for proper form in DOWN position
                if 35 <= elbow_angle <= 45:
                    if side == 'left':
                        left_arm_down = True
                    else:
                        right_arm_down = True
                else:
                    # If not in proper position and not at target up angle
                    if not (140 <= elbow_angle <= 160) and wrist[1] > shoulder[1]:
                        form_violated = True
                        if elbow_angle <= 30:
                            instruction_message = "RAISE YOUR ELBOW POINT!"
                        else:
                            instruction_message = "BEND ELBOWS TO 40 DEGREES!"
            
            # Display arm status for debugging
            cv2.putText(
                image,
                f'L: {int(left_elbow_angle)}° {"DOWN" if left_arm_down else "UP" if left_arm_at_150 else "MID"}',
                (10, image.shape[0] - 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            cv2.putText(
                image,
                f'R: {int(right_elbow_angle)}° {"DOWN" if right_arm_down else "UP" if right_arm_at_150 else "MID"}',
                (10, image.shape[0] - 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            # Track the shoulder press movement using both arms
            if left_arm_down and right_arm_down:
                if stage != "down":
                    print("Setting stage to DOWN")
                stage = "down"
            elif left_arm_at_150 and right_arm_at_150 and stage == "down":
                counter += 1
                stage = "up"
                print(f"Counter increased! Count: {counter}")
            
            # Always display warning message when elbow angle is too low
            if low_elbow_angle:
                warning_message = "RAISE YOUR ELBOW POINT!"
                text_size = cv2.getTextSize(warning_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = image.shape[0] // 2
                
                # Draw semi-transparent background
                overlay = image.copy()
                cv2.rectangle(overlay, 
                            (text_x - 10, text_y - text_size[1] - 10),
                            (text_x + text_size[0] + 10, text_y + 10),
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                
                # Draw text
                cv2.putText(
                    image, 
                    warning_message, 
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2, 
                    cv2.LINE_AA
                )
                
                # Play sound alert if not already playing
                if not sound_playing:
                    sound.play()
                    sound_playing = True
            
            # Handle other form violations
            elif form_violated and not sound_playing:
                sound.play()
                sound_playing = True
                
                # Add instruction message with background for better visibility
                if instruction_message:
                    text_size = cv2.getTextSize(instruction_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (image.shape[1] - text_size[0]) // 2
                    text_y = image.shape[0] // 2
                    
                    # Draw semi-transparent background
                    overlay = image.copy()
                    cv2.rectangle(overlay, 
                                (text_x - 10, text_y - text_size[1] - 10),
                                (text_x + text_size[0] + 10, text_y + 10),
                                (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                    
                    # Draw text
                    cv2.putText(
                        image, 
                        instruction_message, 
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 255), 
                        2, 
                        cv2.LINE_AA
                    )
            # Stop sound if no violations
            elif not form_violated and not low_elbow_angle and sound_playing:
                sound.stop()
                sound_playing = False
            
            # Display counter and stage
            cv2.putText(
                image, 
                f'Count: {counter}',
                (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_AA
            )
            
            cv2.putText(
                image, 
                f'Stage: {stage}',
                (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_AA
            )
            
            # Add target angle indicators
            cv2.putText(
                image,
                "Down: 40° | Up: 150°",
                (10, image.shape[0] - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            
            # Add form guidance text
            cv2.putText(
                image,
                "Start with elbows at 40°",
                (10, image.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                image,
                "Press until elbows reach 150°",
                (10, image.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        # Convert image to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        # Yield frame for Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



# Side Lateral Rasise  /////////////////////////////////////////////////
def side_lateral_raise():
    left_counter = 0
    right_counter = 0
    left_state = "down"
    right_state = "down"
    cap = cv2.VideoCapture(0)
    sound_playing = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            arm_sides = {
                'left': {
                    'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                    'hip': mp_pose.PoseLandmark.LEFT_HIP
                },
                'right': {
                    'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                    'hip': mp_pose.PoseLandmark.RIGHT_HIP
                }
            }

            arm_angle_violated = False
            instruction_message = ""

            for side, joints in arm_sides.items():
                shoulder = landmarks[joints['shoulder'].value]
                elbow = landmarks[joints['elbow'].value]
                wrist = landmarks[joints['wrist'].value]
                hip = landmarks[joints['hip'].value]

                shoulder_coords = (int(shoulder.x * image.shape[1]), int(shoulder.y * image.shape[0]))
                elbow_coords = (int(elbow.x * image.shape[1]), int(elbow.y * image.shape[0]))
                wrist_coords = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
                hip_coords = (int(hip.x * image.shape[1]), int(hip.y * image.shape[0]))

                # Draw connections
                cv2.line(image, shoulder_coords, elbow_coords, (0, 255, 0), 2)
                cv2.line(image, elbow_coords, wrist_coords, (0, 255, 0), 2)
                cv2.line(image, hip_coords, shoulder_coords, (0, 255, 0), 2)

                # Draw joints
                for point in [shoulder_coords, elbow_coords, wrist_coords, hip_coords]:
                    cv2.circle(image, point, 7, (0, 0, 255), -1)

                # Calculate angles
                elbow_angle = calculate_angle([shoulder.x, shoulder.y], [elbow.x, elbow.y], [wrist.x, wrist.y])
                
                # For side lateral raise, we need to measure the angle between hip, shoulder, and elbow
                shoulder_angle = calculate_angle([hip.x, hip.y], [shoulder.x, shoulder.y], [elbow.x, elbow.y])
                
                # Display angles
                cv2.putText(image, f'E: {int(elbow_angle)}', elbow_coords, 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, f'S: {int(shoulder_angle)}', shoulder_coords, 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # We'll simplify the lateral check - removed as it may be causing problems with rep counting
                # Instead we'll focus purely on the angle measurement
                is_lateral = True  # Always true to avoid blocking rep counting
                
                # Draw a visual indicator showing the target angle of 85 degrees
                # This helps the user see where they need to raise their arm to
                target_angle_rad = math.radians(85)
                target_line_length = 100  # pixels
                
                # Calculate end point for the target angle line
                if side == 'left':
                    target_x = shoulder_coords[0] - target_line_length * math.sin(target_angle_rad)
                    target_y = shoulder_coords[1] - target_line_length * math.cos(target_angle_rad)
                else:  # right side
                    target_x = shoulder_coords[0] + target_line_length * math.sin(target_angle_rad)
                    target_y = shoulder_coords[1] - target_line_length * math.cos(target_angle_rad)
                
                # Draw dotted line showing target angle
                target_point = (int(target_x), int(target_y))
                cv2.line(image, shoulder_coords, target_point, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, "85°", target_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Check if shoulder angle exceeds maximum (110 degrees for lateral raise)
                max_shoulder_angle = 110
                if shoulder_angle > max_shoulder_angle:
                    arm_angle_violated = True
                    instruction_message = "LOWER YOUR ARMS!"
                    # Highlight the angle in red to indicate violation
                    cv2.putText(image, f'S: {int(shoulder_angle)}', shoulder_coords, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Check for proper elbow angle (alert if too bent - 100 degrees or less)
                if elbow_angle <= 100:
                    arm_angle_violated = True
                    instruction_message = "STRAIGHTEN YOUR ELBOWS SLIGHTLY!"
                    cv2.putText(image, f'E: {int(elbow_angle)}', elbow_coords, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Simplified and more reliable rep counting
                # Print debug info
                if side == 'left':
                    # Debug print to console - not visible on screen but helps troubleshooting
                    if shoulder_angle >= 85:
                        print(f"Left arm at {shoulder_angle} degrees - State: {left_state}")
                    
                    # DOWN state detection - arms at sides
                    if shoulder_angle < 20 and left_state == "up":
                        left_state = "down"
                        print(f"Left arm DOWN detected - angle: {shoulder_angle}")
                    
                    # UP state detection - arms raised to target angle
                    elif shoulder_angle >= 85 and left_state == "down":
                        left_state = "up"
                        left_counter += 1
                        print(f"Left arm rep counted! Total: {left_counter}")

                elif side == 'right':
                    # Debug print to console
                    if shoulder_angle >= 85:
                        print(f"Right arm at {shoulder_angle} degrees - State: {right_state}")
                    
                    # DOWN state detection - arms at sides
                    if shoulder_angle < 20 and right_state == "up":
                        right_state = "down"
                        print(f"Right arm DOWN detected - angle: {shoulder_angle}")
                    
                    # UP state detection - arms raised to target angle
                    elif shoulder_angle >= 85 and right_state == "down":
                        right_state = "up"
                        right_counter += 1
                        print(f"Right arm rep counted! Total: {right_counter}")
                
                # Display current state on image for debugging
                state_text = "up" if (side == "left" and left_state == "up") or (side == "right" and right_state == "up") else "down"
                cv2.putText(image, f'{side} state: {state_text}', 
                         (int(shoulder.x * image.shape[1]), int(shoulder.y * image.shape[0] - 30)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Control the sound alert
            if arm_angle_violated and not sound_playing:
                sound.play()
                sound_playing = True
            elif not arm_angle_violated and sound_playing:
                sound.stop()
                sound_playing = False
            
            # Display instruction message whenever there's a violation
            if sound_playing and instruction_message:
                # Add centered text with background for better visibility
                text_size = cv2.getTextSize(instruction_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = image.shape[0] // 2
                
                # Draw semi-transparent background for text
                overlay = image.copy()
                cv2.rectangle(overlay, 
                             (text_x - 10, text_y - text_size[1] - 10),
                             (text_x + text_size[0] + 10, text_y + 10),
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                
                # Draw text
                cv2.putText(image, instruction_message, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Display counters and guidance
            cv2.putText(image, f'Left Counter: {left_counter}', (10, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Right Counter: {right_counter}', (10, 100), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Add form guidance with updated angle targets
            cv2.putText(image, f"Left state: {left_state} | Right state: {right_state}", (10, image.shape[0] - 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, "Raise arms laterally to 85 degrees", (10, image.shape[0] - 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Maximum shoulder angle: 110 degrees", (10, image.shape[0] - 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Keep elbows above 100 degrees", (10, image.shape[0] - 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Convert to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Triceps Kickback /////////////////////////////////////////////////////
def triceps_kickback_side():
    """
    Tracks triceps kickback exercise from a side view, similar to the reference image.
    This version focuses on tracking a single arm (the one visible from the side).
    """
    counter = 0
    state = "down"
    cap = cv2.VideoCapture(0)
    sound_playing = False
    
    print("Side View Triceps Kickback exercise started")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # No need to flip frame for side view
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        form_violated = False
        instruction_message = ""
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # For side view, we'll focus on the side that's visible to the camera
            # We'll check which shoulder is more visible/confident and use that side
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            # Determine which side is more visible based on visibility score
            side = 'left' if left_shoulder.visibility > right_shoulder.visibility else 'right'
            
            # Get the landmarks for the selected side
            if side == 'left':
                shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            else:
                shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Convert to coordinate lists for angle calculation
            shoulder_point = [shoulder.x, shoulder.y]
            elbow_point = [elbow.x, elbow.y]
            wrist_point = [wrist.x, wrist.y]
            hip_point = [hip.x, hip.y]
            
            # Convert to pixel coordinates for drawing
            shoulder_coords = (int(shoulder.x * image.shape[1]), int(shoulder.y * image.shape[0]))
            elbow_coords = (int(elbow.x * image.shape[1]), int(elbow.y * image.shape[0]))
            wrist_coords = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
            hip_coords = (int(hip.x * image.shape[1]), int(hip.y * image.shape[0]))
            
            # Draw arm lines and connections
            cv2.line(image, shoulder_coords, elbow_coords, (0, 255, 0), 2)
            cv2.line(image, elbow_coords, wrist_coords, (0, 255, 0), 2)
            cv2.line(image, shoulder_coords, hip_coords, (0, 255, 0), 2)
            
            # Draw joint circles
            for point in [shoulder_coords, elbow_coords, wrist_coords, hip_coords]:
                cv2.circle(image, point, 7, (0, 0, 255), -1)
            
            # Calculate angles
            # 1. Elbow angle: between shoulder-elbow-wrist
            elbow_angle = calculate_angle(shoulder_point, elbow_point, wrist_point)
            
            # 2. Upper arm angle: between hip-shoulder-elbow
            # For side view, this checks if upper arm is parallel to floor
            upper_arm_angle = calculate_angle(hip_point, shoulder_point, elbow_point)
            
            # Print debug info
            print(f"Side: {side}, Elbow angle: {int(elbow_angle)}, Upper arm angle: {int(upper_arm_angle)}")
            
            # Check if torso is bent forward as in the reference image
            # We'll use the angle between vertical and the line from hip to shoulder
            # A value around 45 degrees would indicate proper bent-over position
            
            # First, create a vertical reference point above the hip
            vertical_point = [hip_point[0], hip_point[1] - 0.2]  # Point directly above hip
            torso_angle = calculate_angle(vertical_point, hip_point, shoulder_point)
            
            # Display angles
            cv2.putText(image, f'Elbow: {int(elbow_angle)}°', elbow_coords, 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image, f'Upper arm: {int(upper_arm_angle)}°', 
                      (shoulder_coords[0], shoulder_coords[1] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image, f'Torso: {int(torso_angle)}°', 
                      (hip_coords[0], hip_coords[1] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Check form violations
            
            # 1. Check if upper arm is at 40 degrees or less to trigger alert
            if upper_arm_angle <= 40:
                form_violated = True
                instruction_message = "RAISE YOUR UPPER ARM! ANGLE TOO LOW!"
                cv2.putText(image, f'Upper arm: {int(upper_arm_angle)}°', 
                          (shoulder_coords[0], shoulder_coords[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 2. Check if torso is properly bent forward (30-60 degrees from vertical)
            # This matches the posture shown in the reference image
            if torso_angle < 30 or torso_angle > 60:
                form_violated = True
                instruction_message = "BEND TORSO FORWARD PROPERLY!"
                cv2.putText(image, f'Torso: {int(torso_angle)}°', 
                          (hip_coords[0], hip_coords[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Track exercise state and count reps
            # Only count reps when upper arm angle is above 40 degrees
            if upper_arm_angle > 40 and 30 <= torso_angle <= 60:
                if elbow_angle < 100 and state == "up":
                    state = "down"
                    print(f"DOWN position detected - Elbow angle: {elbow_angle}")
                elif elbow_angle > 150 and state == "down":
                    state = "up"
                    counter += 1
                    print(f"Rep counted! Total: {counter}")
            
            # Control sound alert for form issues
            if form_violated and not sound_playing:
                sound.play()
                sound_playing = True
            elif not form_violated and sound_playing:
                sound.stop()
                sound_playing = False
            
            # Display instruction message when form is violated
            if sound_playing and instruction_message:
                # Add centered text with background for better visibility
                text_size = cv2.getTextSize(instruction_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = image.shape[0] // 2
                
                # Draw semi-transparent background for text
                overlay = image.copy()
                cv2.rectangle(overlay, 
                             (text_x - 10, text_y - text_size[1] - 10),
                             (text_x + text_size[0] + 10, text_y + 10),
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                
                # Draw text
                cv2.putText(image, instruction_message, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Display current state and counter
            cv2.putText(image, f'State: {state.upper()}', (10, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Counter: {counter}', (10, 100), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Add reference image description
            cv2.putText(image, "Side view - Triceps Kickback", (10, image.shape[0] - 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add form guidance with updated angle requirements
            cv2.putText(image, "Bend torso forward 45°", (10, image.shape[0] - 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Keep upper arm ABOVE 40° (Alert at 40° or less)", (10, image.shape[0] - 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Extend arm backward fully", (10, image.shape[0] - 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Convert to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



# Push ups  ////////////////////////////////////////////////////
def push_ups():
    """
    دالة لتتبع وتصحيح تمرين الضغط (Push-ups)
    """
    counter = 0  # عداد التكرارات
    state = None  # حالة التمرين
    cap = cv2.VideoCapture(0)
    sound_playing = False  # حالة تشغيل الصوت
    
    print("Push-ups Exercise Started")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # قلب الإطار أفقياً
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        form_violated = False
        instruction_message = ""
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # تحديد نقاط مهمة للذراعين والجسم
            arm_sides = {
                'left': {
                    'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                    'hip': mp_pose.PoseLandmark.LEFT_HIP,
                    'knee': mp_pose.PoseLandmark.LEFT_KNEE
                },
                'right': {
                    'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                    'hip': mp_pose.PoseLandmark.RIGHT_HIP,
                    'knee': mp_pose.PoseLandmark.RIGHT_KNEE
                }
            }
            
            # حساب زاوية الجسم الكلي
            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            ]
            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]
            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]
            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]
            
            # حساب زاوية الجسم الإجمالية
            body_midpoint_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, 
                                      (left_shoulder[1] + right_shoulder[1])/2]
            body_midpoint_hip = [(left_hip[0] + right_hip[0])/2, 
                                 (left_hip[1] + right_hip[1])/2]
            
            # نقطة رأسية فوق نقطة الوسط
            vertical_point = [body_midpoint_shoulder[0], body_midpoint_shoulder[1] - 0.2]
            
            # زاوية الجسم
            body_angle = calculate_angle(vertical_point, body_midpoint_shoulder, body_midpoint_hip)
            
            # متغيرات لتتبع حالة الذراعين
            left_arm_state = "up"
            right_arm_state = "up"
            
            # معالجة كل ذراع
            for side, joints in arm_sides.items():
                # الحصول على إحداثيات المفاصل
                shoulder = [
                    landmarks[joints['shoulder'].value].x,
                    landmarks[joints['shoulder'].value].y
                ]
                elbow = [
                    landmarks[joints['elbow'].value].x,
                    landmarks[joints['elbow'].value].y
                ]
                wrist = [
                    landmarks[joints['wrist'].value].x,
                    landmarks[joints['wrist'].value].y
                ]
                hip = [
                    landmarks[joints['hip'].value].x,
                    landmarks[joints['hip'].value].y
                ]
                
                # تحويل الإحداثيات إلى إحداثيات الصورة
                shoulder_coords = (int(shoulder[0] * image.shape[1]), int(shoulder[1] * image.shape[0]))
                elbow_coords = (int(elbow[0] * image.shape[1]), int(elbow[1] * image.shape[0]))
                wrist_coords = (int(wrist[0] * image.shape[1]), int(wrist[1] * image.shape[0]))
                hip_coords = (int(hip[0] * image.shape[1]), int(hip[1] * image.shape[0]))
                
                # رسم الخطوط والنقاط
                cv2.line(image, shoulder_coords, elbow_coords, (0, 255, 0), 2)
                cv2.line(image, elbow_coords, wrist_coords, (0, 255, 0), 2)
                cv2.line(image, shoulder_coords, hip_coords, (0, 255, 0), 2)
                
                cv2.circle(image, shoulder_coords, 7, (0, 0, 255), -1)
                cv2.circle(image, elbow_coords, 7, (0, 0, 255), -1)
                cv2.circle(image, wrist_coords, 7, (0, 0, 255), -1)
                cv2.circle(image, hip_coords, 7, (0, 0, 255), -1)
                
                # حساب الزوايا
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_angle = calculate_angle(hip, shoulder, elbow)
                
                # عرض الزوايا
                cv2.putText(image, f'Elbow: {int(elbow_angle)}°', elbow_coords, 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Shoulder: {int(shoulder_angle)}°', shoulder_coords, 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # التحقق من صحة الأداء
                # زاوية المرفق يجب أن تكون حوالي 90 درجة عند النزول
                # زاوية المرفق يجب أن تكون قريبة من 180 درجة عند الرفع
                if elbow_angle < 130:  # نزول
                    if side == 'left':
                        left_arm_state = "down"
                    else:
                        right_arm_state = "down"
                    
                    # التحقق من زاوية الجسم
                    if body_angle > 20:  # انحناء الجسم أكثر من 20 درجة
                        form_violated = True
                        instruction_message = "KEEP YOUR BODY STRAIGHT!"
                        
                elif elbow_angle > 170:  # رفع
                    if side == 'left':
                        left_arm_state = "up"
                    else:
                        right_arm_state = "up"
            
            # منطق حساب التكرارات
            if left_arm_state == "down" and right_arm_state == "down":
                state = "down"
            elif left_arm_state == "up" and right_arm_state == "up" and state == "down":
                counter += 1
                state = "up"
                print(f"Push-up counted! Total: {counter}")
            
            # التحكم في الصوت والتنبيهات
            if form_violated and not sound_playing:
                sound.play()
                sound_playing = True
            elif not form_violated and sound_playing:
                sound.stop()
                sound_playing = False
            
            # عرض رسالة التعليمات
            if sound_playing and instruction_message:
                text_size = cv2.getTextSize(instruction_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = image.shape[0] // 2
                
                # رسم خلفية شبه شفافة
                overlay = image.copy()
                cv2.rectangle(overlay, 
                             (text_x - 10, text_y - text_size[1] - 10),
                             (text_x + text_size[0] + 10, text_y + 10),
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                
                # رسم النص
                cv2.putText(image, instruction_message, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # عرض العداد
            cv2.putText(image, f'Push-ups: {counter}', (10, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # عرض توجيهات إضافية
            cv2.putText(image, "Keep body straight", (10, image.shape[0] - 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Elbows at 90° when down", (10, image.shape[0] - 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Full extension at top", (10, image.shape[0] - 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # تحويل الصورة
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        # إرسال الإطار
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')








# the Lungs ///////////////////////////////////


def lunges():
    left_counter = 0
    right_counter = 0
    left_stage = None
    right_stage = None
    cap = cv2.VideoCapture(0)
    sound_playing = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        form_violated = False
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get leg landmarks for both legs
            leg_sides = {
                'left': {
                    'hip': mp_pose.PoseLandmark.LEFT_HIP,
                    'knee': mp_pose.PoseLandmark.LEFT_KNEE,
                    'ankle': mp_pose.PoseLandmark.LEFT_ANKLE
                },
                'right': {
                    'hip': mp_pose.PoseLandmark.RIGHT_HIP,
                    'knee': mp_pose.PoseLandmark.RIGHT_KNEE,
                    'ankle': mp_pose.PoseLandmark.RIGHT_ANKLE
                }
            }
            
            # Get coordinates for shoulders to check torso alignment
            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            ]
            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]
            
            # Draw line between shoulders
            cv2.line(
                image,
                (int(left_shoulder[0] * image.shape[1]), int(left_shoulder[1] * image.shape[0])),
                (int(right_shoulder[0] * image.shape[1]), int(right_shoulder[1] * image.shape[0])),
                (0, 255, 255),  # Yellow
                2
            )
            
            # Process each leg
            for side, joints in leg_sides.items():
                # Get coordinates for each joint
                hip = [
                    landmarks[joints['hip'].value].x,
                    landmarks[joints['hip'].value].y
                ]
                knee = [
                    landmarks[joints['knee'].value].x,
                    landmarks[joints['knee'].value].y
                ]
                ankle = [
                    landmarks[joints['ankle'].value].x,
                    landmarks[joints['ankle'].value].y
                ]
                
                # Convert to pixel coordinates
                hip_coords = (int(hip[0] * image.shape[1]), int(hip[1] * image.shape[0]))
                knee_coords = (int(knee[0] * image.shape[1]), int(knee[1] * image.shape[0]))
                ankle_coords = (int(ankle[0] * image.shape[1]), int(ankle[1] * image.shape[0]))
                
                # Draw leg lines
                cv2.line(image, hip_coords, knee_coords, (0, 255, 0), 2)
                cv2.line(image, knee_coords, ankle_coords, (0, 255, 0), 2)
                
                # Draw joint circles
                cv2.circle(image, hip_coords, 7, (0, 0, 255), -1)
                cv2.circle(image, knee_coords, 7, (0, 0, 255), -1)
                cv2.circle(image, ankle_coords, 7, (0, 0, 255), -1)
                
                # Calculate knee angle
                knee_angle = calculate_angle(hip, knee, ankle)
                
                # Display knee angle
                cv2.putText(
                    image,
                    f'{int(knee_angle)}°',
                    knee_coords,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Check if knee angle is too small (knee too bent, risk of injury)
                if knee_angle < 70:
                    form_violated = True
                    # Highlight angle in red
                    cv2.putText(
                        image,
                        f'{int(knee_angle)}°',
                        knee_coords,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),  # Red
                        2,
                        cv2.LINE_AA
                    )
                
                # Calculate hip height and track lunge state
                if side == 'left':
                    # Check if left leg is in lunge position (knee bent)
                    if knee_angle < 120 and hip[1] > knee[1]:
                        left_stage = "down"
                    # Check if returned to standing position
                    elif knee_angle > 160 and left_stage == "down":
                        left_stage = "up"
                        left_counter += 1
                        print(f'Left Lunge Counter: {left_counter}')
                
                elif side == 'right':
                    # Check if right leg is in lunge position (knee bent)
                    if knee_angle < 120 and hip[1] > knee[1]:
                        right_stage = "down"
                    # Check if returned to standing position
                    elif knee_angle > 160 and right_stage == "down":
                        right_stage = "up"
                        right_counter += 1
                        print(f'Right Lunge Counter: {right_counter}')
            
            # Check torso alignment
            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]
            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]
            
            # Draw line between hips
            cv2.line(
                image,
                (int(left_hip[0] * image.shape[1]), int(left_hip[1] * image.shape[0])),
                (int(right_hip[0] * image.shape[1]), int(right_hip[1] * image.shape[0])),
                (0, 255, 255),  # Yellow
                2
            )
            
            # Calculate torso angle (vertical alignment)
            left_torso_point = [(left_shoulder[0] + left_hip[0])/2, (left_shoulder[1] + left_hip[1])/2]
            right_torso_point = [(right_shoulder[0] + right_hip[0])/2, (right_shoulder[1] + right_hip[1])/2]
            
            torso_angle = abs(90 - calculate_angle(
                [left_torso_point[0], 0],  # Point directly above
                left_torso_point,
                right_torso_point
            ))
            
            # Display torso angle
            torso_point = (
                int((left_torso_point[0] + right_torso_point[0])/2 * image.shape[1]),
                int((left_torso_point[1] + right_torso_point[1])/2 * image.shape[0])
            )
            cv2.putText(
                image,
                f'Torso: {int(torso_angle)}°',
                (torso_point[0], torso_point[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            # Check if torso is leaning too far (not upright)
            if torso_angle > 20:
                form_violated = True
                cv2.putText(
                    image,
                    f'Torso: {int(torso_angle)}°',
                    (torso_point[0], torso_point[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),  # Red
                    2,
                    cv2.LINE_AA
                )
            
            # Play sound alert if form is violated
            if form_violated and not sound_playing:
                sound.play()
                sound_playing = True
                
                # Add instruction message with background for better visibility
                text = "FIX YOUR FORM!"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = image.shape[0] // 2
                
                # Draw semi-transparent background
                overlay = image.copy()
                cv2.rectangle(overlay, 
                             (text_x - 10, text_y - text_size[1] - 10),
                             (text_x + text_size[0] + 10, text_y + 10),
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                
                # Draw text
                cv2.putText(
                    image, 
                    text, 
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2, 
                    cv2.LINE_AA
                )
            elif not form_violated and sound_playing:
                sound.stop()
                sound_playing = False
            
            # Display counters
            cv2.putText(
                image, 
                f'Left Lunges: {left_counter}', 
                (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_AA
            )
            cv2.putText(
                image, 
                f'Right Lunges: {right_counter}', 
                (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_AA
            )
        
        # Convert image to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        # Yield frame for Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# //////////////////////////////////////////////////Blank exercise /////////////////////////////////////////////////
def plank():
    """
    دالة لتتبع تمرين البلانك وحساب المدة والتأكد من صحة الوضعية
    
    العائد:
        مولد إطارات الفيديو المعالجة للعرض عبر Flask
    """
    cap = cv2.VideoCapture(0)
    plank_start_time = None
    plank_duration = 0
    correct_posture = False
    sound_playing = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # قلب الإطار أفقياً
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # الحصول على إحداثيات النقاط المهمة للبلانك
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            
            # تحويل الإحداثيات النسبية إلى إحداثيات الصورة
            left_shoulder_coords = (int(left_shoulder[0] * image.shape[1]), int(left_shoulder[1] * image.shape[0]))
            right_shoulder_coords = (int(right_shoulder[0] * image.shape[1]), int(right_shoulder[1] * image.shape[0]))
            left_hip_coords = (int(left_hip[0] * image.shape[1]), int(left_hip[1] * image.shape[0]))
            right_hip_coords = (int(right_hip[0] * image.shape[1]), int(right_hip[1] * image.shape[0]))
            left_ankle_coords = (int(left_ankle[0] * image.shape[1]), int(left_ankle[1] * image.shape[0]))
            right_ankle_coords = (int(right_ankle[0] * image.shape[1]), int(right_ankle[1] * image.shape[0]))
            left_knee_coords = (int(left_knee[0] * image.shape[1]), int(left_knee[1] * image.shape[0]))
            right_knee_coords = (int(right_knee[0] * image.shape[1]), int(right_knee[1] * image.shape[0]))
            
            # رسم الخطوط والنقاط المهمة
            # رسم خط الجسم
            cv2.line(image, left_shoulder_coords, left_hip_coords, (0, 255, 0), 2)
            cv2.line(image, right_shoulder_coords, right_hip_coords, (0, 255, 0), 2)
            cv2.line(image, left_hip_coords, left_knee_coords, (0, 255, 0), 2)
            cv2.line(image, right_hip_coords, right_knee_coords, (0, 255, 0), 2)
            cv2.line(image, left_knee_coords, left_ankle_coords, (0, 255, 0), 2)
            cv2.line(image, right_knee_coords, right_ankle_coords, (0, 255, 0), 2)
            cv2.line(image, left_shoulder_coords, right_shoulder_coords, (0, 255, 255), 2)
            cv2.line(image, left_hip_coords, right_hip_coords, (0, 255, 255), 2)
            
            # رسم النقاط المهمة
            for point in [left_shoulder_coords, right_shoulder_coords, left_hip_coords, right_hip_coords, 
                          left_ankle_coords, right_ankle_coords, left_knee_coords, right_knee_coords]:
                cv2.circle(image, point, 7, (0, 0, 255), -1)
            
            # حساب الزوايا المهمة للتحقق من صحة وضعية البلانك
            # زاوية الجسم (الكتف - الورك - الكاحل)
            left_body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
            right_body_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
            
            # زاوية الكوع (الكتف - الكوع - الرسغ) - في حالة البلانك الاعتيادي غير مهمة
            
            # زاوية الركبة (الورك - الركبة - الكاحل)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            # عرض الزوايا على الصورة
            cv2.putText(image, f'Body Angle L: {int(left_body_angle)}', (10, 150), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Body Angle R: {int(right_body_angle)}', (10, 180),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Knee Angle L: {int(left_knee_angle)}', (10, 210),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Knee Angle R: {int(right_knee_angle)}', (10, 240),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # التحقق من صحة وضعية البلانك
            # في البلانك الصحيح: زاوية الجسم يجب أن تكون قريبة من 180 درجة (مستقيمة)
            # وزاوية الركبة قريبة من 180 درجة أيضاً (ساقين ممدودتين)
            
            # ضبط عتبات الزوايا المثالية
            body_angle_min = 160  # الحد الأدنى لزاوية استقامة الجسم
            knee_angle_min = 160  # الحد الأدنى لزاوية استقامة الركبة
            
            # التحقق من الوضعية
            if (left_body_angle > body_angle_min and right_body_angle > body_angle_min and
                left_knee_angle > knee_angle_min and right_knee_angle > knee_angle_min):
                correct_posture = True
                # بدء احتساب الوقت إذا لم نكن قد بدأنا بالفعل
                if plank_start_time is None:
                    plank_start_time = cv2.getTickCount()
                
                # إيقاف صوت التنبيه إذا كان يعمل
                if sound_playing:
                    sound.stop()
                    sound_playing = False
                    
                # حساب المدة
                current_time = cv2.getTickCount()
                elapsed_time = (current_time - plank_start_time) / cv2.getTickFrequency()
                plank_duration = elapsed_time
                
            else:
                correct_posture = False
                # إعادة ضبط وقت البداية عند انقطاع الوضعية الصحيحة
                plank_start_time = None
                
                # تشغيل صوت التنبيه إذا لم يكن يعمل بالفعل
                if not sound_playing:
                    sound.play()
                    sound_playing = True
            
            # عرض حالة الوضعية والوقت
            status_text = "Correct Posture" if correct_posture else "Incorrect Posture"
            status_color = (0, 255, 0) if correct_posture else (0, 0, 255)  # أخضر للصحيح، أحمر للخاطئ
            
            cv2.putText(image, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
            
            # عرض الوقت فقط إذا كانت الوضعية صحيحة
            if correct_posture:
                minutes = int(plank_duration // 60)
                seconds = int(plank_duration % 60)
                cv2.putText(image, f'Time: {minutes:02d}:{seconds:02d}', (10, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # تحويل الصورة إلى تنسيق JPEG للبث
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        # إعادة الإطار لاستجابة Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

def triceps_extension():
    counter = 0  
    state = None 
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            
            arm_sides = {
                'right': {
                    'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.RIGHT_WRIST
                },
                'left': {
                    'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.LEFT_WRIST
                }
            }

            for side, joints in arm_sides.items():
                shoulder = [
                    landmarks[joints['shoulder'].value].x,
                    landmarks[joints['shoulder'].value].y,
                ]
                elbow = [
                    landmarks[joints['elbow'].value].x,
                    landmarks[joints['elbow'].value].y,
                ]
                wrist = [
                    landmarks[joints['wrist'].value].x,
                    landmarks[joints['wrist'].value].y,
                ]

                
                shoulder_coords = (int(shoulder[0] * image.shape[1]), int(shoulder[1] * image.shape[0]))
                elbow_coords = (int(elbow[0] * image.shape[1]), int(elbow[1] * image.shape[0]))
                wrist_coords = (int(wrist[0] * image.shape[1]), int(wrist[1] * image.shape[0]))

               
                cv2.line(image, shoulder_coords, elbow_coords, (0, 255, 0), 2)
                cv2.line(image, elbow_coords, wrist_coords, (0, 255, 0), 2)

                
                cv2.circle(image, shoulder_coords, 7, (0, 0, 255), -1)
                cv2.circle(image, elbow_coords, 7, (0, 0, 255), -1)
                cv2.circle(image, wrist_coords, 7, (0, 0, 255), -1)

                
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

               
                cv2.putText(
                    image,
                    f'{int(elbow_angle)}°',
                    elbow_coords,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                if elbow_angle < 45:
                    state = "down"
                if elbow_angle > 160 and state == "down":
                    state = "up"
                    counter += 1
                    print(f'Triceps Reps: {counter}')

            
            cv2.putText(image, f'Triceps Reps: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<exercise>')
def video_feed(exercise):
    if exercise == 'hummer':
        return Response(hummer(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif exercise == 'front_raise':
        return Response(dumbbell_front_raise(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif exercise == 'squat':
        return Response(squat(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif exercise == 'triceps':
        return Response(triceps_extension(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif exercise == 'lunges':
        return Response(lunges(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif exercise == 'shoulder_press':
        return Response(shoulder_press(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif exercise == 'plank':
        return Response(plank(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif exercise == 'side_lateral_raise':
        return Response(side_lateral_raise(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif exercise == 'triceps_kickback_side':
        return Response(triceps_kickback_side(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif exercise == 'push_ups':
        return Response(push_ups(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid exercise", 400

if __name__ == '__main__':
    app.run(debug=True)




@app.route('/api/pose_data')
def pose_data():
    # Get the pose data and return it as a JSON response
    data = hummer()
    return data

if __name__ == '__main__':
    app.run(debug=True)