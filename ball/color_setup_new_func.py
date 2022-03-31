import time
from tkinter import *
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
import numpy as np 
import os
# import eventlet
# os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from pygame import mixer

import models
from detection.face import get_largest_frame

app = Tk()
app.title('ITrain Crop')
app.geometry('500x700')
app.attributes('-topmost',True)

app.tk.call("source", "static/theme/azure.tcl")
app.tk.call("set_theme", "light")

title = ttk.Label(app, text='Crop the ball', font='arial 30 bold', foreground='#000')
title.pack()

IMG_HEIGHT = 480
IMG_WIDTH = 640

# Declaration of variables
mask = np.ones((IMG_HEIGHT, IMG_WIDTH))
minute=StringVar()
second=StringVar()
minute.set("00")
second.set("03")

image_area = Canvas(app, width=IMG_WIDTH, height=IMG_HEIGHT, bg='#C8C8C8')
image_area.pack(pady=(10,0))

minuteEntry= ttk.Entry(app, width=3, font=("Arial",18,""),
                textvariable=minute)
minuteEntry.place(x=210,y=280)
secondEntry= ttk.Entry(app, width=3, font=("Arial",18,""),
                textvariable=second)
secondEntry.place(x=260,y=280)

mixer.init()


def show_mask():
    global image_for_mask_multiplication
    global cropped_image, cropped_image_cv2
    mask_3_channels = np.ones((IMG_HEIGHT, IMG_WIDTH, 3)) 
    image_mattt = (mask * 255).astype(np.uint8)
    the_real_mask = return_shape(image_mattt)
    mask_3_channels[:,:,0] = the_real_mask/255
    mask_3_channels[:,:,1] = the_real_mask/255
    mask_3_channels[:,:,2] = the_real_mask/255
    real_area = np.array(image_for_mask_multiplication) * mask_3_channels
    real_area = Image.fromarray(np.uint8(real_area)).convert('RGB')
    
    cropped_image = real_area.convert("RGB") # RGBA
    datas = cropped_image.getdata()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    cropped_image.putdata(newData)
    cropped_image_cv2 = np.array(cropped_image) 
    # Convert RGB to BGR 
    cropped_image_cv2 = cropped_image_cv2[:, :, ::-1].copy()
    process()
    # cropped_image.show()


show_area = ttk.Button(app, width=20, text='Done', command=show_mask)
show_area.pack(pady=(0,5))

lower = np.array([0, 0, 0])
higher = np.array([255, 255, 255])

lower_green = np.array([0, 0, 0])
higher_green = np.array([255, 255, 255])
lower_red = np.array([0, 0, 0])
higher_red = np.array([255, 255, 255])

    # save_image = Button(app, width=20, text='SAVE IMAGE', font='none 12', command=save_image)
    # save_image.pack()

def reset_color():    
    global app, title, mask, minute, second, image_area, minuteEntry, secondEntry
    global show_area, lower, higher

    app = Tk()
    app.title('Crop')
    app.geometry('500x700')
    app.attributes('-topmost',True)
    title = Label(app, text='Crop the ball', font='arial 30 bold', fg='#068481')
    title.pack()

    # Declaration of variables
    mask = np.ones((IMG_HEIGHT, IMG_WIDTH))
    minute=StringVar()
    second=StringVar()
    minute.set("00")
    second.set("03")

    image_area = Canvas(app, width=IMG_WIDTH, height=IMG_HEIGHT, bg='#C8C8C8')
    image_area.pack(pady=(10,0))

    minuteEntry= Entry(app, width=3, font=("Arial",18,""),
                    textvariable=minute)
    minuteEntry.place(x=210,y=280)
    secondEntry= Entry(app, width=3, font=("Arial",18,""),
                    textvariable=second)
    secondEntry.place(x=260,y=280)

    mixer.init()

    show_area = ttk.Button(app, width=20, text='Done', command=show_mask)
    show_area.pack(pady=(0,5))

    lower = np.array([0, 0, 0])
    higher = np.array([255, 255, 255])

def run_color():
    submit()
    app.mainloop()

    # app.destroy()
    print("Done!")
    # global lower, higher
    global lower_green, higher_green, lower_red, higher_red

    return [lower_green, higher_green, lower_red, higher_red]
 
def blur_img( img, factor = 20):
    # kW = int(img.shape[1] / factor)
    # kH = int(img.shape[0] / factor)
    # #ensure the shape of the kernel is odd
    # if kW % 2 == 0: kW = kW - 1
    # if kH % 2 == 0: kH = kH - 1
    # blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
    
    blurred_img = np.zeros([img.shape[0],img.shape[1],3],dtype=np.uint8)
    blurred_img.fill(255)
    return blurred_img

def takePhoto():
    global image, image_for_mask_multiplication, original_image
    global image_area, app

    vidcap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    _,image = vidcap.read()
    blurred_img = blur_img(image, factor = 2)
    faces = models.face_cascade.detectMultiScale(image, 1.1, 4)
    faces = get_largest_frame(faces)
    blur_margin = 0
    for (x, y, w, h) in faces:
        detected_face = image[int(y)-blur_margin:, int(x)-blur_margin:int(x+w)+blur_margin]
        blurred_img[y-blur_margin:, x-blur_margin:x+w+blur_margin] = detected_face
        image = blurred_img

        original_image = detected_face

    if models.is_frozen:
        cv2.imwrite(os.path.join(models.EXE_LOCATION,'data','reference_color.jpg'), image)
    else:
        cv2.imwrite(os.path.join(models.EXE_LOCATION,'data','reference_color.jpg'), image)
    
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = Image.fromarray(image)
    image_for_mask_multiplication = image
    image = ImageTk.PhotoImage(image)
    image_area.create_image(0, 0, image=image, anchor='nw')
    select_area()
    
def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y

def draw_smth( event):
    global lasx, lasy
    image_area.create_line((lasx, lasy, event.x, event.y), fill='red', width=3)
    lasx, lasy = event.x, event.y
    
    if lasx < 640 and lasx >=0 and lasy < 480 and lasy >=0:
        mask[lasy][lasx] = 0 
        mask[lasy+1][lasx+1] = 0 
        mask[lasy-1][lasx-1] = 0 
        mask[lasy+1][lasx-1] = 0 
        mask[lasy-1][lasx+1] = 0 

def select_area():
    global image_area

    image_area.bind("<Button-1>", get_x_and_y)
    image_area.bind("<B1-Motion>", draw_smth)

def return_shape(image_in):
    image = image_in
    gray = image_in
    edged = cv2.Canny(gray, 30, 200) 
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    cv2.drawContours(image, contours, -1, (0, 0, 0), 3)  
    th, im_th = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), (255,255,255))
    im_floodfill = np.abs(im_floodfill-np.ones((IMG_HEIGHT, IMG_WIDTH))*255)
    return im_floodfill
    #cv2.imshow("Floodfilled Image", im_floodfill)


def remove_white(img):
    ## (1) Convert to gray, and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    
    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # print(cnts)
    cnt = sorted(cnts, key=cv2.contourArea) #[-1]
    green_cnt = cnt[-1]
    red_cnt = cnt[-2]
    
    ## (4) Crop and save it
    x,y,w,h = cv2.boundingRect(green_cnt)
    dst_green = img[y:y+h, x:x+w]

    x,y,w,h = cv2.boundingRect(red_cnt)
    dst_red = img[y:y+h, x:x+w]
    return [dst_green, dst_red]

def find_range(cropped_image_cv2):
    hsv = cv2.cvtColor(cropped_image_cv2, cv2.COLOR_BGR2HSV)

    height, width, channels = hsv.shape
    lower = np.array([255,255,255])
    higher = np.array([0,0,0])
    # crop = np.asarray(cropped_image)
    # print(crop)
    for y in range(0, height):
        for x in range(0, width):
            pixel = hsv[y,x]
            if (not np.array_equal(pixel, np.array([255,255,255]))
                and not np.array_equal(pixel, np.array([0,0,0]))):
                lower[0] = min(pixel[0], lower[0]) if pixel[0] != 0 else lower[0]
                lower[1] = min(pixel[1], lower[1]) if pixel[1] != 0 else lower[1]
                lower[2] = min(pixel[2], lower[2]) if pixel[2] != 0 else lower[2]
                
                higher[0] = max(pixel[0],higher[0]) if pixel[0] != 255 else higher[0]
                higher[1] = max(pixel[1],higher[1]) if pixel[1] != 255 else higher[1]
                higher[2] = max(pixel[2],higher[2]) if pixel[2] != 255 else higher[2]
                
    print(lower)
    print(higher)

    return [lower, higher]

def process():
    global cropped_image, cropped_image_cv2, original_image
    # global lower, higher
    global lower_green, higher_green, lower_red, higher_red
    
    cv2.imwrite("0.jpg",cropped_image_cv2)

    # cropped_image_cv2 = remove_white(cropped_image_cv2)
    green, red = remove_white(cropped_image_cv2)

    lower_green, higher_green = find_range(green)
    lower_red, higher_red = find_range(red)

    # original_image_hsv = hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # mask = cv2.inRange(original_image_hsv, lower, higher)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    # cv2.imwrite(os.path.join(models.EXE_LOCATION,'data','11.jpg'), mask)

    app.destroy()
    # return [lower, higher]


def submit():
    temp = int(minute.get())*60 + int(second.get())     
    while temp >-1:
        mins,secs = divmod(temp,60)
        minute.set("{0:2d}".format(mins))
        second.set("{0:2d}".format(secs))   
        app.update()

        mixer.music.load("static/audio/beep.wav")
        mixer.music.play()

        time.sleep(1)   
        # eventlet.sleep(1)
        if (temp == 0):
            minuteEntry.destroy()
            secondEntry.destroy()
            takePhoto()
            # messagebox.showinfo("Time Countdown", "Time's up ")
        temp -= 1

