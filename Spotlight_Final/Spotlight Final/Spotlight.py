#################################################################
# Name: Mosa Alsalih, Angel Beltran, and Anthony Karam
# Date: 04/16/2022
# Description: Final Pi Project Spotlight
# Resources:
#
#################################################################

# import the necessary libraries
from djitellopy import tello
import pygame
import cv2 as cv
import button
from face_recognition import face_encodings, face_locations, compare_faces, face_distance
from os import path, listdir
import numpy
import time
import KeyboardCommands as KeyC

# code that will allow the drone to be controlled from a keyboard
def keyboardInput(me):
    global land
    # goes up to 100cm/s in speed
    # positive is right, forward, up, and yall right
    # negative number is left, back, down, yall left
    # creating a list is the most efficient way we found our code to work
    # we are setting the initial movement of our drone to 0 so it maintains stationary in the air
    lr, fb, up, yv = 0, 0, 0, 0
    speed = 60

    # left key
    if KeyC.getKey("LEFT"):
        lr = -speed
    # right key
    elif KeyC.getKey("RIGHT"):
        lr = speed
    # up key
    if KeyC.getKey("UP"):
        fb = speed
    # down key
    elif KeyC.getKey("DOWN"):
        fb = -speed
    # w key goes up
    if KeyC.getKey("w"):
        up = speed
    # s key goes down
    elif KeyC.getKey("s"):
        up = -speed
    # d key yall right
    if KeyC.getKey("d"):
        yv = speed
    # a key yalls left
    elif KeyC.getKey("a"):
        yv = -speed
    # r key lands
    if KeyC.getKey("r"):
        me.land()
        land = True
    # t key takes off
    if KeyC.getKey("t"):
        me.takeoff()
        land = False
    # returns list of controls in respect to keys pressed
    return [lr, fb, up, yv]

# encoding function takes the images in the faces folder and makes a 128 dimensional face encoding(an array)
# keep in mind that each face in the folder is being encoded and appended into the encoding list.
def encoding(faces):
    encodelist = []
    for img in faces:
        # rgb color scale
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # encoding process
        encode = face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

# this is our attendance function which documents known and unknown faces into a txt file
def attendance(name):
    # getting current time in the format Apr, Thur, 14, 10:46:seconds
    current_time = time.strftime("%b, %a, %d, %I:%M:%S %p")
    # opens a txt file and appends the strings below depending on the met condition
    with open("attendance.txt", 'a') as file_object:
        if name == "Unknown":
            file_object.write(f"{name} individual was spotted at {current_time}\n")
        else:
            file_object.write(f"{name} is being recognized at {current_time}\n")

# code that will track and recognize faces, and then update the drone screen on the gui
def recognition(myFrame):
    global btime, counter, ak, ab, ma, recognized

    # initial setting
    recognized = False
    ak = False
    ab = False
    ma = False

    # resizing the camera window
    img = cv.resize(myFrame, (576, 432))
    resize = cv.resize(img, (0, 0), None, 0.25, 0.25)

    # outputting fps text onto the camera window
    atime = time.time()
    fps = round(1 / (atime - btime), 2)
    btime = atime
    # prints frames per second
    cv.putText(img, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    # prints battery life
    cv.putText(img, f"Battery: {me.get_battery()}%", (10, 55), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    # prints temperature
    cv.putText(img, f"Temp: {me.get_temperature()}F", (10, 80), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # initial number of faces
    numFaces = 0

    # looking for faces, upsample increases the distance of recognition
    currentfaces = face_locations(resize, number_of_times_to_upsample=2)
    # encodes the faces that are being recognized
    currentencodefaces = face_encodings(resize, currentfaces, num_jitters=1)

    # double for loop of the variables created above
    for encodeface, facelocation in zip(currentencodefaces, currentfaces):
        # comparing the encoding that we made in our encoding function which are our known to the current encoding of facial recognition
        match = compare_faces(known, encodeface, tolerance=0.6)
        # comparing the geometry of the images
        facedis = face_distance(known, encodeface)
        # turning it into an array
        matchIndext = numpy.argmin(facedis)

        # face location gets the top, right, bottom, left of a face when in search for one in the recognition process
        # from thses numbers we are able to create a bounding box around the face of a recognized face
        y1, x2, y2, x1 = facelocation
        # since we resized the camera screen we have to multiply by 4 to get the right scale for the bounding box
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        # drawing bounding box around the face
        cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # creating a small box that will have the names written on them
        cv.rectangle(img, (x1, y2 - 25), (x2, y2), (255, 0, 0), cv.FILLED)

        # should be from 0.30 to 0.75
        # 72 to 216
        # the name with adjust to the distance with this scale
        scale = (x2 - x1) / 72
        if scale >= 3:
            scale = 2.5

        # if our face comparisons match with the geometry array it will right the name down of the person being recognized
        if match[matchIndext]:
            name = names[matchIndext]
            # writes name on the bounding box
            cv.putText(img, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 0.30 * scale, (255, 255, 255), 2)
            # used for extra information screen
            # if recognized it will help make the button appear for that certain person
            recognized = True
            if name == "Anthony_Karam":
                ak = True
            if name == "Angel_Beltran":
                ab = True
            if name == "Mosa_Alsalih":
                ma = True
            # calling attendance function
            attendance(name)
        # when a documented face is not found it will be name unknown with a face bounding box
        else:
            cv.putText(img, "Unknown", (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 0.30 * scale, (255, 255, 255), 2)
            # calling attendance using unknown as the parameter
            attendance("Unknown")
        # counting number of faces being recognized
        numFaces += 1
    # drawing the number of faces on the camera screen
    cv.putText(img, f"Number of Faces: {numFaces}", (10, 105), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # flipping drone camera
    img = cv.flip(img, 1)
    # rgb color scale
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = numpy.rot90(img)
    img = pygame.surfarray.make_surface(img)
    screen.blit(img, (0, 0))

# code that will set up the elements necessary for the start screen
def createStartScreen():
    global display_button, takeoff_button, controls_button, exit_button

    # screen.fill('yellow')
    # We are setting images on top of each other
    screen.blit(starting_screen, (0, 0))
    screen.blit(title, title.get_rect(center=(400, 50)))

    awayFromSide = (SCREEN_WIDTH - (display_img.get_width()+takeoff_img.get_width())) * (5/12)
    # these lines of code are setting and positioning of our buttons
    display_button = button.Button(SCREEN_WIDTH-(awayFromSide+display_img.get_width()), 275, display_img)
    # setting takeoff button
    takeoff_button = button.Button(awayFromSide, 275, takeoff_img)
    # setting controls button
    controls_button = button.Button((SCREEN_WIDTH-(controls_img.get_width()))/2, 375, controls_img)
    # setting exit button
    exit_button = button.Button(SCREEN_WIDTH-(exit_img.get_width()*0.6), SCREEN_HEIGHT-(exit_img.get_height()*0.6), exit_img, 0.6)

# code that will set up the elements necessary for the drone screen
def createDroneScreen():
    global exit_button, land_button, takeoff_button, homescreen_button, ak_button, ab_button, ma_button
    # putting drone screen on top of our image logo
    screen.blit(starting_screen, (0, 0))
    # positioning of buttons on Gui when we have the drone camera running
    exit_button = button.Button(SCREEN_WIDTH - (exit_img.get_width() * 0.6) + 1, SCREEN_HEIGHT - (exit_img.get_height() * 0.6) + 1, exit_img, 0.6)
    land_button = button.Button(576 - (land_img.get_width() * 0.6), SCREEN_HEIGHT - (land_img.get_height() * 0.6) + 1, land_img, 0.6)
    takeoff_button = button.Button((0 * 0.6), SCREEN_HEIGHT - (takeoff_img.get_height() * 0.6) + 1, takeoff_img, 0.6)
    homescreen_button = button.Button((takeoff_img.get_width() * 0.6), SCREEN_HEIGHT - (homescreen_img.get_height() * 0.6) + 1, homescreen_img, 0.6)
    # extra information screen buttons for our team members
    ak_button = button.Button(SCREEN_WIDTH - (ak_img.get_width()) + 1, 20, ak_img)
    ab_button = button.Button(SCREEN_WIDTH - (ab_img.get_width()) + 1, 20 + ak_img.get_height(), ab_img)
    ma_button = button.Button(SCREEN_WIDTH - (ma_img.get_width()) + 1, 20 + ak_img.get_height() + ab_img.get_height(), ma_img)

# code that will set up the elements necessary for the controls screen
def createControlsScreen():
    global homescreen_button
    # # filling background color
    screen.fill((125, 255, 243))
    # keys images are put on top of background screen
    screen.blit(key_image, key_image.get_rect(center=(395, 250)))
    # home screen button
    homescreen_button = button.Button((0 * 0.6), SCREEN_HEIGHT - (homescreen_img.get_height() * 0.6) + 1, homescreen_img, 0.6)

# code that will set up the elements necessary for the extra information screen
def createExtraInformationScreen(name):
    global exit_button, display_button
    # our team member info is put on top of any previous screen
    screen.blit(name, (0, 0))
    # button to go back to display screen
    display_button = button.Button(SCREEN_WIDTH - (display_img.get_width() * 0.6), SCREEN_HEIGHT - (exit_img.get_height() * 0.6) - (display_img.get_height() * 0.6) + 1, display_img, 0.6)
    # exit button to exit program
    exit_button = button.Button(SCREEN_WIDTH - (exit_img.get_width() * 0.6) + 1, SCREEN_HEIGHT - (exit_img.get_height() * 0.6) + 1, exit_img, 0.6)

# code that will continuously run when the current screen displayed is the start screen
def start_screen():
    global land
    # drone will takeoff and display drone camera view
    if takeoff_button.draw(screen):
        land = False
        # loads drone screen
        createDroneScreen()
        # drones camera on
        me.streamon()
        # drone takeoff
        me.takeoff()
        return False, True, False, False, True
    # Will terminate the program
    elif exit_button.draw(screen):
        return False, False, False, False, False
    # will display the drone camera view
    elif display_button.draw(screen):
        print("Display Button Clicked: " + str(time.time()-start))
        me.streamon()
        createDroneScreen()
        return False, True, False, False, True
    # will load the instructions/ controls for the drone screen
    elif controls_button.draw(screen):
        createControlsScreen()
        return False, False, True, False, True
    # remains the start screen
    else:
        return True, False, False, False, True

# code that will continuously run when the current screen displayed is the drone screen
def drone_screen():
    global land, camera, recognized, ak, ma, ab, counter

    frame_read = me.get_frame_read()
    myFrame = frame_read.frame
    # running facial recognition
    recognition(myFrame)

    if counter == 0:
        print("Display Button Process Complete: " + str(time.time()-start))
        counter += 1

    screen.blit(starting_screen_right, (SCREEN_WIDTH - starting_screen_right.get_width(), 0))
    # if one of our team members is recognized a button will appear on our GUI with the member's name
    # if the button is pressed an extra info screen will pop up showing info on that team member
    if recognized:
        if ma and ma_button.draw(screen):
            createExtraInformationScreen(ma_info)
            return False, False, False, True, True
        if ak and ak_button.draw(screen):
            createExtraInformationScreen(ak_info)
            return False, False, False, True, True
        if ab and ab_button.draw(screen):
            createExtraInformationScreen(ab_info)
            return False, False, False, True, True
    # when the land button is pressed it will display an exit, takeoff, and homescreen button
    # drone camera view is still on as well as facial recognition
    if land:
        if KeyC.getKey("r"):
            me.land()
            land = True
        # when landed you can take off again using t key or the GUI
        if KeyC.getKey("t"):
            me.takeoff()
            land = False
        # exit button onto GUI
        if exit_button.draw(screen):
            land = True
            return False, False, False, False, False
        # setting takeoff button
        elif takeoff_button.draw(screen):
            land = False
            me.takeoff()
            screen.blit(starting_screen_bottom, (0, SCREEN_HEIGHT - starting_screen_bottom.get_height()))
            return False, True, False, False, True
        # homescreen button
        elif homescreen_button.draw(screen):
            me.streamoff()
            createStartScreen()
            return True, False, False, False, True
        else:
            return False, True, False, False, True
    # these are the buttons that remain when drone takes off if no action occurs
    else:
        # setting values in respect to key board input
        vals = keyboardInput(me)
        # rc controls for drone
        me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
        # exit button
        if exit_button.draw(screen):
            land = True
            me.land()
            return False, False, False, False, False
        # land button
        elif land_button.draw(screen):
            land = True
            me.land()
            screen.blit(starting_screen_bottom, (0, SCREEN_HEIGHT - starting_screen_bottom.get_height()))
            return False, True, False, False, True
        else:
            return False, True, False, False, True

# code that will continuously run when the current screen displayed is the controls screen
def controls_screen():
    # if home screen is pressed it will return to home
    if homescreen_button.draw(screen):
        createStartScreen()
        return True, False, False, False, True
    # if nothing is pressed, will remain in controls screen
    else:
        return False, False, True, False, True

# code that will continuously run when the current screen displayed is the extra information screen
def extraInformation_screen():
    global land
    # when our program runs we have a small screen of the drone camera view on the top right corner
    frame_read = me.get_frame_read()

    img = frame_read.frame
    # resizing image for the top right corner
    img = cv.resize(img, (96, 72))
    # flipping camera
    img = cv.flip(img, 1)
    # RGB color scale
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # rotates array in 90 degrees to get pixels of images
    img = numpy.rot90(img)
    img = pygame.surfarray.make_surface(img)
    screen.blit(img, (SCREEN_WIDTH - 96, 0))
    # taking keyboard input for land and takeoff
    if land:
        # r key for landing
        if KeyC.getKey("r"):
            me.land()
            land = True
        # t key for takeoff
        if KeyC.getKey("t"):
            me.takeoff()
            land = False
    # if extra info screen is pressed drone will continue to take flight controls
    else:
        # continues ro take input from key board
        vals = keyboardInput(me)
        me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    # display button
    if display_button.draw(screen):
        # creates drone screen
        createDroneScreen()
        return False, True, False, False, True
    # exits program
    elif exit_button.draw(screen):
        return False, False, False, False, False
    return False, False, False, True, True

#create display window
start = time.time()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 480

pygame.init()
# pygame window dimensions
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Spotlight')

# load button images and all other images used 
starting_screen = pygame.image.load('files/Team3_Logo.jpg').convert_alpha()
starting_screen_bottom = pygame.image.load('files/Team3_LogoBottom.jpg').convert_alpha()
starting_screen_right = pygame.image.load('files/Team3_LogoRight.jpg').convert_alpha()
title = pygame.image.load('files/title.png').convert_alpha()

homescreen_img = pygame.image.load('files/button_homescreen.png').convert_alpha()
display_img = pygame.image.load('files/button_display.png').convert_alpha()
takeoff_img = pygame.image.load('files/button_takeoff.png').convert_alpha()
controls_img = pygame.image.load('files/button_controls.png').convert_alpha()
exit_img = pygame.image.load('files/button_exit.png').convert_alpha()
land_img = pygame.image.load('files/button_land.png').convert_alpha()
key_image = pygame.image.load('files/keyboard.png').convert_alpha()
ak_img = pygame.image.load('files/button_anthonykaram.png').convert_alpha()
ab_img = pygame.image.load('files/button_angelbeltran.png').convert_alpha()
ma_img = pygame.image.load('files/button_mosaalsalih.png').convert_alpha()
ak_info = pygame.image.load('files/ak_info.png').convert_alpha()
ab_info = pygame.image.load('files/ab_info.png').convert_alpha()
ma_info = pygame.image.load('files/ma_info.png').convert_alpha()
print("Load all images: " + str(time.time() - start))

# putting all images in faces folder into a list
directory = 'faces'
faces = []
names = []
list = listdir(directory)
for x in range(len(list)):
    if list[x] == '.DS_Store':
        list.pop(x)
        break
print(list)

# with this for loop we are simply taking out the .jpg
for n in list:
    curimg = cv.imread(f"{directory}/{n}")
    faces.append(curimg)
    names.append(path.splitext(n)[0])
print(names)

# known faces encoding for later comparisons
known = encoding(faces)
# once encoding presses is complete check
print('Encoding complete')

# create button instances
createStartScreen()

# initial state of program so only start screen
run = True
startScreen = True
droneScreen = False
controlsScreen = False
extraInformationScreen = False
recognized = False

# encoding procces time
print("Encode: " + str(time.time() - start))

# connecting to tello drone and turning on camera
# gathering fps
me = tello.Tello()
# connecting to drone
me.connect()
# turning on drone camera
me.streamon()
print((time.time() - start))
frame_read = me.get_frame_read()
print((time.time() - start))
frame_read = me.get_frame_read()
print((time.time() - start))
myFrame = frame_read.frame
print((time.time() - start))
# turning off drone camera
me.streamoff()

land = True

counter = 0
btime = 0

print("Homescreen Launch: " + str(time.time() - start))

# Program loop all have a boolean value
while run:

    if startScreen:
        startScreen, droneScreen, controlsScreen, extraInformationScreen, run = start_screen()
    elif droneScreen:
        startScreen, droneScreen, controlsScreen, extraInformationScreen, run = drone_screen()
    elif controlsScreen:
        startScreen, droneScreen, controlsScreen, extraInformationScreen, run = controls_screen()
    elif extraInformationScreen:
        startScreen, droneScreen, controlsScreen, extraInformationScreen, run = extraInformation_screen()

    # event handler
    for event in pygame.event.get():
        # quit game
        if event.type == pygame.QUIT:
            run = False
    pygame.display.update()

print("Goodbye")
# prints battery life
print(me.get_battery())
# camera screen shuts off
cv.destroyAllWindows()
pygame.quit()