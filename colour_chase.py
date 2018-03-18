import asyncio
import cozmo
import PIL.Image
import PIL.ImageFont
import PIL.ImageTk
import PIL.Image
import cv2
import numpy as np
import math
import tkinter

from cozmo.util import degrees


'''
Colour Chase
- Experimenting with Cozmo's camera, async operation,  OpenCV & tkinter diplay and control.
- Displays tkinter window with cozmo's camera image appended with identified areas.
- Includes Tkinter buttons for Mode selection and drive control
- Calibrate samples a displayed area in the centre of the image with other modes 
controlled by the colour range set here. 

Modes are 

    1.  View/STOP   Displays camera views and STOPS Cozmo movement (starting Mode).
                    Whilst View Mode can be run on its own it is ALWAYS active showing
                    Cozmo's camera image.
    
    2.  Calibrate   Creates Colour Threshold levels based on colour samples from the
                    centre of images colour.
                    Tolerance levels are added to try and cope with differing light angles. 
                    Currently set to (h_tol = 2, s_tol = 40, v_tol = 30) tight hue tolerance with 
                    looser saturation and value. 
    
    3.  Chase       Cozmo will look around foe calibrated colour object - when seen head for it.
                    If object is moved he will Chase it otherwise it will push it around.
                    This Mode has no End and can be terminated by selecting View/STOP Mode or by using Drive Controls.
    
    4.  PLay        This is chase with the extra condition that after every fourth drive cozmo puts 
                    in a 90 degree turn. This is meant to keep him within a tighter area so that he 
                    can go on playing with the object himself (indefinitely).     
    
    5.  Fetch       In thi mode Cozmo looks around for colour - heads for it - goes around it 
                    and tries to pus it back to starting position position when mode selected).
                    If successful stops (program returns to View/STOP mode    
    
    Modes can be repeatedly run by button selection.
    
    In addition, a set of controls to drive Cozmo are included. The drive forward, turn left 
    and turn right options have 3 speed settings. Repeatedly hitting the appropriate button 
    will speed the drive up to max and then reduce back to slowest setting.
    
    Program can be terminated by closing the tkinter window by clicking 'X'

author Jthomm based on structure and various code elements from 
Threshold test (@author Team Cozplay)
'''

MODE_VIEW = 'view'
MODE_CALIBRATE = 'calibrate'
MODE_CHASE = 'chase'
MODE_FETCH = 'fetch'
MODE_PLAY = 'play'
MODE_QUIT = 'close_all'

DRIVE_FORWARD = 'forwards'
DRIVE_BACKWARD = 'backwards'
TURN_LEFT = 'turn_left'
TURN_RIGHT = 'turn_right'
DRIVE_STOP = 'stop'

# Calibration window & number of frames to sample
WINDOW_DIM = 40     # calibration sample window - (dim x dim)
NUM_FRAMES = 20
WINDOW_OFFSET = int(WINDOW_DIM/2)

# Cozmo's image is 320 x 240
X_WIDTH = 320
Y_HEIGHT = 240
CENTRE_X = int(X_WIDTH/2)
CENTRE_Y = int(Y_HEIGHT/2)


class ColourChase:
    def __init__(self):
        self._robot = None
        self._tk_root = 0
        self._tk_img_output = 0
        self._tk_select_mode = 0
        self._tk_view = 0
        self._tk_calibrate = 0
        self._tk_search = 0
        self._tk_play = 0
        self._tk_fetch = 0

        self._tk_coz_drive = 0
        self._tk_drive_f = 0
        self._tk_drive_b = 0
        self._tk_turn_l = 0
        self._tk_turn_r = 0
        self._tk_drive_s = 0

        self.w = 0 # width of tk window
        self.h = 0 # height of tk window

        self.df_speed = 0
        self.tl_angle = 0
        self.tr_angle = 0

        self.mode = MODE_VIEW

        self.calibrate_count = 0
        self.calibrate_size = NUM_FRAMES * WINDOW_DIM * WINDOW_DIM
        self.calibrate = np.empty((self.calibrate_size, 3), int)

        self.fetch_start = True
        self.turn_complete = False
        self.turn_to_object_attempts = 0
        self.play_count = 0

        # START WITH None as defaultCOLOUR
        self.low_colour = np.array([0, 0, 0], int)
        self.high_colour = np.array([0, 0, 0], int)
        self.lower_calibrate = np.array([0, 0, 0], int)
        self.upper_calibrate = np.array([0, 0, 0], int)
        self.hue_rotate = 0

        self.object_centre = np.array([0, 0], int)
        self.object_size = np.array([0, 0], int)
        self.coz_x_init = 0
        self.coz_y_init = 0

        cozmo.connect(self.run)

    def callback_close(self):
        # Detect the tk window close request by USER clicking 'X'
        print('Close')
        self.mode = MODE_QUIT

    def callback_view(self):
        print('View')
        self.clear_drive_rates()
        self.mode = MODE_VIEW

    def callback_calibrate(self):
        print('Calibrate')
        self.clear_drive_rates()
        self.calibrate_count = 0
        self.calibrate_size = NUM_FRAMES * WINDOW_DIM * WINDOW_DIM
        self.calibrate = np.empty((self.calibrate_size, 3), int)
        self.mode = MODE_CALIBRATE

    def callback_chase(self):
        print('Chase')
        self.clear_drive_rates()
        self.turn_to_object_attempts = 0
        self.mode = MODE_CHASE

    def callback_play(self):
        print('Play')
        self.clear_drive_rates()
        self.clear_drive_rates()
        self.turn_to_object_attempts = 0
        self.play_count = 0
        self.mode = MODE_PLAY

    def callback_fetch(self):
        print('Fetch')
        self.clear_drive_rates()
        self.fetch_start = True
        self.turn_complete = False
        self.turn_to_object_attempts = 0
        self.mode = MODE_FETCH

    def callback_df(self):
        print('Drive Forward')
        self.tl_angle = 0
        self.tr_angle=0
        if self.df_speed == 3:
            self.df_speed = 1
        else:
            self.df_speed +=1
        self.mode = DRIVE_FORWARD

    def callback_db(self):
        print('Drive Back')
        self.clear_drive_rates()
        self.mode = DRIVE_BACKWARD

    def callback_tl(self):
        print('Turn Left')
        self.df_speed = 0
        self.tr_angle = 0
        if self.tl_angle == 3:
            self.tl_angle = 1
        else:
            self.tl_angle += 1
        self.mode = TURN_LEFT

    def callback_tr(self):
        print('Turn Right')
        self.df_speed = 0
        self.tl_angle = 0
        if self.tr_angle == 3:
            self.tr_angle = 1
        else:
            self.tr_angle    += 1
        self.mode = TURN_RIGHT

    def callback_ds(self):
        print('Stop')
        self.clear_drive_rates()
        self.mode = DRIVE_STOP

    def clear_drive_rates(self):
        self.df_speed = 0
        self.tl_angle = 0
        self.tr_angle = 0

    def on_new_camera_image(self, event, *, image: cozmo.world.CameraImage, **kw):
        cv_yellow = (0, 255, 255)

        raw_image = image.raw_image
        # Convert PIL Image to OpenCV Image
        imgBGR = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)

        # convert brg to hsv
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
        # Calibrate object in central part of images Colour
        if self.mode == MODE_CALIBRATE:
            self.calibrate_object(imgHSV)

        # if threshold was rotated do the same with image
        if self.hue_rotate > 0:
            imgHSV[:, :, 0] = (imgHSV[:, :, 0] + self.hue_rotate) % 180

        # Create binary image using calibrated colour range
        imgThresh = cv2.inRange(imgHSV, self.low_colour, self.high_colour)

        # Clean image
        # 1st erode - dilate (remove noise and restore form
        kernelOpen = np.ones((5, 5))
        imgOpen = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernelOpen)
        # 2nd dilate - erode (fill gaps, then reduce to original size
        kernelClose = np.ones((40, 40))
        imgClose = cv2.morphologyEx(imgOpen, cv2.MORPH_CLOSE, kernelClose)

        if self.mode == MODE_CALIBRATE:
            cv2.rectangle(imgBGR, (CENTRE_X - WINDOW_OFFSET, CENTRE_Y - WINDOW_OFFSET), (CENTRE_X + WINDOW_OFFSET, CENTRE_Y + WINDOW_OFFSET), cv_yellow, 2)
            cv2.putText(imgBGR, 'CALIBRATING', (CENTRE_X - WINDOW_DIM, CENTRE_Y + WINDOW_DIM), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv_yellow, 1, cv2.LINE_AA)
        else:
            # create contours around remaining binary shapes
            img_copy, contours, hierarchy = cv2.findContours(imgClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # select largest contoured shape
            size = 0
            big_cont = 0
            for i in range(len(contours)):
                x, y, w, h = cv2.boundingRect(contours[i])
                if w*h > size:
                    size = w*h
                    big_cont = i
            self.object_centre = np.array([0, 0], int)
            self.object_size = np.array([0, 0], int)
            # turn contour into rectangle and draw rectangle on image
            if len(contours) > 0:
                x, y, w, h = cv2.boundingRect(contours[big_cont])
                cv2.rectangle(imgBGR, (x, y), (x + w, y + h), cv_yellow, 2)
                self.object_centre = np.array([x+w/2, y+h/2])
                self.object_size = np.array([w, h])

        b_dim = 20      # Drive buttons are 20 x 20
        # Drive button images are scaled to reflect speed (or rate of turn)
        if self.df_speed == 1:
            im_dim = 12
        else:
            if self.df_speed == 2:
                im_dim = 16
            else:
                im_dim = 20
        df_img = PIL.Image.open('drive_img/df.GIF')
        df_img = df_img.resize((im_dim, im_dim), PIL.Image.NEAREST)
        dfk_img = PIL.ImageTk.PhotoImage(df_img)
        self._tk_drive_f.configure(image=dfk_img, width=b_dim, height=b_dim, command=self.callback_df, bd=5, bg='light pink')

        db_img = PIL.Image.open('drive_img/db.GIF')
        db_img = db_img.resize((b_dim, b_dim), PIL.Image.NEAREST)
        dbk_img = PIL.ImageTk.PhotoImage(db_img)
        self._tk_drive_b.configure(image=dbk_img, width=b_dim, height=b_dim, command=self.callback_db, bd=5, bg='light pink')

        if self.tl_angle == 1:
            im_dim = 12
        else:
            if self.tl_angle == 2:
                im_dim = 16
            else:
                im_dim = 20
        tl_img = PIL.Image.open('drive_img/tl.GIF')
        tl_img = tl_img.resize((im_dim, im_dim), PIL.Image.NEAREST)
        tlk_img = PIL.ImageTk.PhotoImage(tl_img)
        self._tk_turn_l.configure(image=tlk_img, width=b_dim, height=b_dim, command=self.callback_tl, bd=5, bg='light pink')

        if self.tr_angle == 1:
            im_dim = 12
        else:
            if self.tr_angle == 2:
                im_dim = 16
            else:
                im_dim = 20
        tr_img = PIL.Image.open('drive_img/tr.GIF')
        tr_img = tr_img.resize((im_dim, im_dim), PIL.Image.NEAREST)
        trk_img = PIL.ImageTk.PhotoImage(tr_img)
        self._tk_turn_r.configure(image=trk_img, width=b_dim, height=b_dim, command=self.callback_tr, bd=5, bg='light pink')

        ds_img = PIL.Image.open('drive_img/ds.GIF')
        ds_img = ds_img.resize((b_dim, b_dim), PIL.Image.NEAREST)
        dsk_img = PIL.ImageTk.PhotoImage(ds_img)
        self._tk_drive_s.configure(image=dsk_img, width=b_dim, height=b_dim, command=self.callback_ds, bd=5, bg='light pink')

        label_font = ('Courier', 12, 'bold')
        button_font = ('Courier', 8, 'bold')

        # Rescale & Convert output images back to PIL image
        img_xdim = int(self.w * 0.965)
        img_ydim = int(img_xdim * Y_HEIGHT/X_WIDTH )
        Final_Pil = PIL.Image.fromarray(cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB))
        Final_Pil = Final_Pil.resize((img_xdim, img_ydim), PIL.Image.ANTIALIAS)
        display_image_output = PIL.ImageTk.PhotoImage(image=Final_Pil)
        self._tk_img_output.configure(image=display_image_output)

        self._tk_coz_drive.configure(text='Drive Cozmo', font=label_font)
        self._tk_select_mode.configure(text='Select Cozmo Mode', font=label_font)
        mode_bx_size = 10
        mode_by_size = 1
        self._tk_view.configure(text='View/STOP', width=mode_bx_size, height=mode_by_size, command=self.callback_view, font=button_font, bd=5, bg='pale turquoise')
        self._tk_calibrate.configure(text='Calibrate', width=mode_bx_size, height=mode_by_size, command=self.callback_calibrate, font=button_font, bd=5, bg='pale turquoise')
        mode_bx_size = 7
        self._tk_search.configure(text='Chase', width=mode_bx_size, height=mode_by_size, command=self.callback_chase, font=button_font, bd=5, bg='pale turquoise')
        self._tk_play.configure(text='Play', width=mode_bx_size, height=mode_by_size, command=self.callback_play, font=button_font, bd=5, bg='pale turquoise')
        self._tk_fetch.configure(text='Fetch', width=mode_bx_size, height=mode_by_size, command=self.callback_fetch, font=button_font, bd=5, bg='pale turquoise')
        self._tk_root.protocol("WM_DELETE_WINDOW", self.callback_close)
        self._tk_root.update()

    def calibrate_object(self, imgHSV):
        # Collect Calibration readings from defined window for selected No. of frames
        if self.calibrate_count < self.calibrate_size:
            for i in range(WINDOW_DIM):
                for j in range(WINDOW_DIM):
                    self.calibrate[self.calibrate_count] = imgHSV[(CENTRE_Y+i, CENTRE_X+j)]
                    self.calibrate_count += 1

        # Analyse Readings to set New calibration levels
        if self.calibrate_count >= self.calibrate_size:
            # Stage 1 : Remove any row with Hue & Saturation = 0 (< 0.1)
            # as early frames (frame 1) on connection seem to have doubtful readings
            cal_index = []
            for i in range(self.calibrate_size):
                if self.calibrate[i][0] < 0.1 and self.calibrate[i][1] < 0.1:
                    cal_index.append(i)

            self.calibrate = np.delete(self.calibrate, (cal_index), axis=0)
            self.calibrate_size -= len(cal_index)

            # Stage 2 : Calculate Average Hue, Saturation and Value
            #  The average Hue is Complicated by circular nature of hue as it
            #  crosses from 179 to 0 degrees
            h_calibrate = 0
            s_calibrate = 0
            v_calibrate = 0

            x_total = 0
            y_total = 0
            for i in range(self.calibrate_size):
                h_test = self.calibrate[i][0]
                h_test = h_test * math.pi / 90  # as h 0 - 180
                x = math.cos(h_test)
                y = math.sin(h_test)
                x_total += x
                y_total += y
            h_calibrate = math.atan2(y_total, x_total)
            h_calibrate = h_calibrate * 90 / math.pi     # as h 0 - 180
            # remove any -ve representation
            if h_calibrate <= 0:
                h_calibrate += 180
            h_calibrate = int(h_calibrate)

            for i in range(self.calibrate_size):
                s_calibrate += self.calibrate[i][1]
                v_calibrate += self.calibrate[i][2]
            s_calibrate = int(s_calibrate / self.calibrate_size)
            v_calibrate = int(v_calibrate / self.calibrate_size)

            # Stage 3 Add tolerance to increase range of h,s, & v to allow for detection
            # at differing light angles
            h_tol = 2
            s_tol = 40
            v_tol = 30
            h_min = h_calibrate - h_tol
            h_max = h_calibrate + h_tol
            s_min = s_calibrate - s_tol
            s_max = s_calibrate + s_tol
            v_min = v_calibrate - v_tol
            v_max = v_calibrate + v_tol

            if h_min < 0:
                h_min += 180
            if h_max > 180:
                h_max -= 180
            if s_min < 0:
                s_min = 0
            if s_max > 255:
                s_max = 255
            if v_min < 0:
                v_min = 0
            if v_max > 255:
                v_max = 255
            self.lower_calibrate = np.array([h_min, s_min, v_min], int)
            self.upper_calibrate = np.array([h_max, s_max, v_max], int)
            print('lower_calibrate = ', self.lower_calibrate)
            print('upper_calibrate = ', self.upper_calibrate)

            # if hue crossed 179 - 0 boundary rotate colour thresholds
            if self.lower_calibrate[0] > self.upper_calibrate[0]:
                self.hue_rotate = 20
                self.lower_calibrate[0] = (self.lower_calibrate[0] + self.hue_rotate) % 180
                self.upper_calibrate[0] = (self.upper_calibrate[0] + self.hue_rotate) % 180
                print('hue rotated')
                print('lower_calibrate = ', self.lower_calibrate)
                print('upper_calibrate = ', self.upper_calibrate)
            else:
                self.hue_rotate = 0

            # Change current Colour Calibration levels
            self.low_colour = self.lower_calibrate
            self.high_colour = self.upper_calibrate

            # Return to view mode - but with NEW calibration detect showing
            self.callback_view()

    async def chase_object(self):
        object_centre_x = self.object_centre[0]
        centre_tol = 12
        drive_time = 1.5
        drive_speed = 150
        turn_speed = 80
        turn_angle = 20

        if object_centre_x == 0:
            await self._robot.turn_in_place(degrees(turn_angle), degrees(turn_speed)).wait_for_completed()
        else:
            if (abs(CENTRE_X - object_centre_x) < centre_tol) or self.turn_to_object_attempts > 3:
                await self._robot.drive_wheels(drive_speed, drive_speed, duration=drive_time)
                self.turn_to_object_attempts = 0
            else:
                turn_to_object = ((CENTRE_X - object_centre_x) / X_WIDTH) * self._robot.camera.config.fov_x.degrees
                await self._robot.turn_in_place(degrees(turn_to_object), degrees(turn_speed)).wait_for_completed()
                self.turn_to_object_attempts += 1

    async def play_with_object(self):
        object_centre_x = self.object_centre[0]
        centre_tol = 12
        drive_time = 1.5
        drive_speed = 120
        turn_speed = 80
        turn_angle = 20

        if object_centre_x == 0:
            await self._robot.turn_in_place(degrees(turn_angle), degrees(turn_speed)).wait_for_completed()
        else:
            if self.play_count == 4:
                self.play_count = 0
                for stage in range(5):
                    if self.mode != MODE_PLAY:
                        break
                    if stage == 0:
                        await self._robot.turn_in_place(degrees(90), degrees(turn_speed)).wait_for_completed()
                    if stage == 1:
                        await self._robot.drive_wheels(100, 100, duration=1.5)
                    if stage == 2:
                        await self._robot.turn_in_place(degrees(-90), degrees(turn_speed)).wait_for_completed()
                    if stage == 3:
                        await self._robot.drive_wheels(100, 100, duration=1.5)
                    if stage == 4:
                        await self._robot.turn_in_place(degrees(-90), degrees(turn_speed)).wait_for_completed()
                object_centre_x = self.object_centre[0]
            if (abs(CENTRE_X - object_centre_x) < centre_tol) or self.turn_to_object_attempts > 3:
                self.play_count += 1
                await self._robot.drive_wheels(drive_speed, drive_speed, duration=drive_time)
                self.turn_to_object_attempts = 0
            else:
                turn_to_object = ((CENTRE_X - object_centre_x) / X_WIDTH) * self._robot.camera.config.fov_x.degrees
                await self._robot.turn_in_place(degrees(turn_to_object), degrees(turn_speed)).wait_for_completed()
                self.turn_to_object_attempts += 1

    async def fetch_object(self):
        object_size_x = self.object_size[0]
        object_centre_x = self.object_centre[0]
        centre_tol = 12
        # slow down when approaching object - stop when too close
        slow_down = 0.15 * X_WIDTH  # 0.2 Tennis Ball - cube size object 0.1 * X_WIDTH
        too_close = 0.3 * X_WIDTH  # 0.4 Tennis Ball - cube size object 0.2 * X_WIDTH
        turn_speed = 100
        turn_angle = 20
        slow_speed = 70
        fast_speed = 150
        return_speed = 160
        drive_time_out = 1.2
        drive_time_return = 1.5
        drive_time = drive_time_out

        # x distance driving back in fetch for SUCCESS
        x_proximity = 180   # x and y axis are set by direction cozmo
        y_proximity = 180    # is facing when fetch is selected.

        # Collect starting pose
        if self.fetch_start:
            # collect starting position
            self.coz_x_init = self._robot.pose.position.x
            self.coz_y_init = self._robot.pose.position.y
            self.fetch_start = False

        if object_centre_x == 0:
            await self._robot.turn_in_place(degrees(turn_angle), degrees(turn_speed)).wait_for_completed()
        else:
            if self.turn_complete:
                drive_time = drive_time_return
                drive_speed = return_speed
            else:
                drive_time = drive_time_out
                if object_size_x > slow_down:
                    drive_speed = slow_speed
                else:
                    drive_speed = fast_speed
            if (abs(CENTRE_X - object_centre_x) < centre_tol) or (self.turn_to_object_attempts > 3):
                await self._robot.drive_wheels(drive_speed, drive_speed, duration=drive_time)
                self.turn_to_object_attempts = 0
            else:
                turn_to_object = ((CENTRE_X - object_centre_x) / X_WIDTH) * self._robot.camera.config.fov_x.degrees
                await self._robot.turn_in_place(degrees(turn_to_object), degrees(turn_speed)).wait_for_completed()
                self.turn_to_object_attempts += 1

            if (not self.turn_complete) and (object_size_x > too_close):
                # drive around ball and face return direction
                for stage in range(7):
                    if self.mode != MODE_FETCH:
                        break
                    if stage == 0:
                        await self._robot.turn_in_place(degrees(90), degrees(turn_speed)).wait_for_completed()
                    if stage == 1:
                        await self._robot.drive_wheels(100, 100, duration=1.5)
                    if stage == 2:
                        await self._robot.turn_in_place(degrees(-90), degrees(turn_speed)).wait_for_completed()
                    if stage == 3:
                        await self._robot.drive_wheels(100, 100, duration=3.0)
                    if stage == 4:
                        await self._robot.turn_in_place(degrees(-90), degrees(turn_speed)).wait_for_completed()
                    if stage == 5:
                        await self._robot.drive_wheels(100, 100, duration=1.5)
                    if stage == 6:
                        await self._robot.turn_in_place(degrees(-90), degrees(turn_speed)).wait_for_completed()
                self.turn_complete = True

            # Collect return pose
            if self.turn_complete:
                # check check position
                coz_x = self._robot.pose.position.x
                coz_y = self._robot.pose.position.y
                x_dist = abs(coz_x - self.coz_x_init)
                y_dist = abs(coz_y - self.coz_y_init)

                if (x_dist < x_proximity) and (y_dist < y_proximity):
                    # Fetch successful, celebrate & return to View Mode
                    await self._robot.say_text('Whoopee !', play_excited_animation=True).wait_for_completed()
                    self.callback_view()

    async def drive_forward(self):
        f_speed = 60 * self.df_speed
        await self._robot.drive_wheels(f_speed, f_speed, duration=0.5)

    async def drive_backward(self):
        await self._robot.drive_wheels(-100, -100, duration=0.5)

    async def turn_left(self):
        t_angle = 10 * self.tl_angle
        await self._robot.turn_in_place(degrees(t_angle ), degrees(80)).wait_for_completed()

    async def turn_right(self):
        t_angle = -10 * self.tr_angle
        await self._robot.turn_in_place(degrees(t_angle), degrees(80)).wait_for_completed()

    async def drive_stp(self):
        await self._robot.drive_wheels(0, 0, duration=0.0)

    async def set_up_cozmo(self, coz_conn):
        asyncio.set_event_loop(coz_conn._loop)
        self._robot = await coz_conn.wait_for_robot()
        self._robot.camera.image_stream_enabled = True
        self._robot.add_event_handler(cozmo.world.EvtNewCameraImage, self.on_new_camera_image)
        self._robot.camera._auto_exposure_enabled = True
        self._robot.camera.color_image_enabled = True
        self._robot.set_head_angle(degrees(-4))

    async def run(self, coz_conn):
        await self.set_up_cozmo(coz_conn)

        self._tk_root = tkinter.Tk()
        self._tk_img_output = tkinter.Label(self._tk_root)
        self._tk_select_mode = tkinter.Label(self._tk_root)
        self._tk_view = tkinter.Button(self._tk_root)
        self._tk_calibrate = tkinter.Button(self._tk_root)
        self._tk_search = tkinter.Button(self._tk_root)
        self._tk_play = tkinter.Button(self._tk_root)
        self._tk_fetch = tkinter.Button(self._tk_root)

        self._tk_coz_drive = tkinter.Label(self._tk_root)
        self._tk_drive_f = tkinter.Button(self._tk_root)
        self._tk_drive_b = tkinter.Button(self._tk_root)
        self._tk_turn_l = tkinter.Button(self._tk_root)
        self._tk_turn_r = tkinter.Button(self._tk_root)
        self._tk_drive_s = tkinter.Button(self._tk_root)

        # Set tk window dimensions & position
        # get the screen size : width & height
        hs = self._tk_root.winfo_screenheight()
        ws = self._tk_root.winfo_screenwidth()
        # set the window (root) size
        self.h = hs * 0.7  # height for the Tk window (as fraction of screen height)
        self.w = self.h * 0.7  #set aspect ratio forthe tk window
        # calculate x & y position for the selected window screen position
        x_pos = ws * 0.7 - self.w / 2    # x position of tk window
        y_pos = hs * 0.5 - self.h / 2    # y position of tk window
        self._tk_root.geometry('%dx%d+%d+%d' % (self.w, self.h, x_pos, y_pos))

        self._tk_img_output.place(x=0.01 * self.w, y=0)
        self._tk_select_mode.place(x=0.26 * self.w, y=0.52 * self.h)
        self._tk_view.place(x=0.2 * self.w, y=0.57 * self.h)
        self._tk_calibrate.place(x=0.6 * self.w, y=0.57 * self.h)
        self._tk_search.place(x=0.1 * self.w, y=0.65 * self.h)
        self._tk_play.place(x=0.4 * self.w, y=0.65 * self.h)
        self._tk_fetch.place(x=0.7 * self.w, y=0.65 * self.h)

        self._tk_coz_drive.place(x=0.33 * self.w, y=0.75 * self.h)
        self._tk_drive_f.place(x=0.45 * self.w, y=0.79 * self.h)
        self._tk_drive_b.place(x=0.45 * self.w, y=0.91 * self.h)
        self._tk_turn_l.place(x=0.55 * self.w, y=0.85 * self.h)
        self._tk_turn_r.place(x=0.36 * self.w, y=0.85 * self.h)
        self._tk_drive_s.place(x=0.45 * self.w, y=0.85 * self.h)
        self._tk_root.attributes('-topmost', True)

        while True:
            await asyncio.sleep(0)
            if self.mode == MODE_CHASE:
                await self.chase_object()
            if self.mode == MODE_PLAY:
                await self.play_with_object()
            if self.mode == MODE_FETCH:
                await self.fetch_object()
            if self.mode == MODE_QUIT:
                self._tk_root.destroy()
                quit()
            if self.mode == DRIVE_FORWARD:
                await self.drive_forward()
            if self.mode == DRIVE_BACKWARD:
                await self.drive_backward()
            if self.mode == TURN_LEFT:
                await self.turn_left()
            if self.mode == TURN_RIGHT:
                await self.turn_right()
            if self.mode == DRIVE_STOP:
                await self.drive_stp()


if __name__ == '__main__':
    ColourChase()
