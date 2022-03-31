import time
from tkinter import *
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np 
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

import models


app = Tk()
app.title('Crop')
app.geometry('500x700')
title = Label(app, text='CROP THE IMAGE', font='arial 30 bold', fg='#068481')
title.pack()

IMG_HEIGHT = 480
IMG_WIDTH = 640


# Declaration of variables
minute=StringVar()
second=StringVar()

mask = np.ones((IMG_HEIGHT, IMG_WIDTH))
global image
global cropped_image, cropped_image_cv2

# setting the default value as 0
minute.set("00")
second.set("03")

minuteEntry= Entry(app, width=3, font=("Arial",18,""),
				textvariable=minute)
minuteEntry.place(x=130,y=20)

secondEntry= Entry(app, width=3, font=("Arial",18,""),
				textvariable=second)
secondEntry.place(x=180,y=20)

image_area = Canvas(app, width=IMG_WIDTH, height=IMG_HEIGHT, bg='#C8C8C8')
image_area.pack(pady=(10,0))

# def openAndPut():
#     path = filedialog.askopenfilename()
#     global image
#     global image_for_mask_multiplication
#     if path:
#         image = Image.open(path)
#         image_for_mask_multiplication = Image.open(path)
#         image = image.resize((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS)
#         image_for_mask_multiplication = image_for_mask_multiplication.resize((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS)
#         image = ImageTk.PhotoImage(image)
#         image_area.create_image(0, 0, image=image, anchor='nw')

def takePhoto():
    # path = filedialog.askopenfilename()
    global image
    global image_for_mask_multiplication

    vidcap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    _,image = vidcap.read()

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

def draw_smth(event):
    global lasx, lasy
    image_area.create_line((lasx, lasy, event.x, event.y), fill='red', width=3)
    lasx, lasy = event.x, event.y

    
    if lasx < 500 and lasx >=0 and lasy < 400 and lasy >=0:
        mask[lasy][lasx] = 0 
        mask[lasy+1][lasx+1] = 0 
        mask[lasy-1][lasx-1] = 0 
        mask[lasy+1][lasx-1] = 0 
        mask[lasy-1][lasx+1] = 0 

def select_area():
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

# def save_image():
#     # path_save = filedialog.asksaveasfilename()
#     # print(path_save)
#     global image, cropped_image, image_for_mask_multiplication
#     print(type(image))
#     print(type(cropped_image))
#     print(type(image_for_mask_multiplication))
#     # if path_save:
#         # image.save(str(path_save), "PNG")
#     cropped_image.save("img.jpg", "JPEG")


def remove_white(img):

    ## (1) Convert to gray, and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    ## (4) Crop and save it
    x,y,w,h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]

    return dst

def process():
    global cropped_image, cropped_image_cv2

    cropped_image_cv2 = remove_white(cropped_image_cv2)
    height, width, channels = cropped_image_cv2.shape

    lower = np.array([255,255,255])
    higher = np.array([0,0,0])
    # crop = np.asarray(cropped_image)
    # print(crop)

    for y in range(0, height):
        for x in range(0, width):
            pixel = cropped_image_cv2[y,x]
            if (not np.array_equal(pixel, np.array([255,255,255]))
                and not np.array_equal(pixel, np.array([0,0,0]))):
                lower[0] = min(pixel[0], lower[0])
                lower[1] = min(pixel[1], lower[1])
                lower[2] = min(pixel[2], lower[2])
                
                higher[0] = max(pixel[0],higher[0])
                higher[1] = max(pixel[1],higher[1])
                higher[2] = max(pixel[2],higher[2])
                # print(cropped_image_cv2[y,x])
            # if cropped_image_cv2[y,x] != 
            # print(cropped_image_cv2[y,x])

    print(lower)
    print(higher)


def submit():
    temp = int(minute.get())*60 + int(second.get())     
    while temp >-1:
        mins,secs = divmod(temp,60)

        minute.set("{0:2d}".format(mins))
        second.set("{0:2d}".format(secs))   
        app.update()
        time.sleep(1)   
        if (temp == 0):
            takePhoto()
        	# messagebox.showinfo("Time Countdown", "Time's up ")

        temp -= 1

submit()

show_area = Button(app, width=20, text='SHOW AREA', font='none 12', command=show_mask)
show_area.pack(pady=(0,5))

# save_image = Button(app, width=20, text='SAVE IMAGE', font='none 12', command=save_image)
# save_image.pack()

app.mainloop()
