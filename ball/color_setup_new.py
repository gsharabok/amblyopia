import time
from tkinter import *
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np 
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2


import models
from detection.face import get_largest_frame

class ColorSetup:
    def __init__(self):
        self.app = Tk()
        self.app.title('Crop')
        self.app.geometry('500x700')
        self.app.attributes('-topmost',True)
        self.title = Label(self.app, text='Crop the ball', font='arial 30 bold', fg='#068481')
        self.title.pack()

        self.IMG_HEIGHT = 480
        self.IMG_WIDTH = 640

        # Declaration of variables
        self.mask = np.ones((self.IMG_HEIGHT, self.IMG_WIDTH))

        self.minute=StringVar()
        self.second=StringVar()
        self.minute.set("00")
        self.second.set("03")

        self.minuteEntry= Entry(self.app, width=3, font=("Arial",18,""),
                        textvariable=self.minute)
        self.minuteEntry.place(x=130,y=20)

        self.secondEntry= Entry(self.app, width=3, font=("Arial",18,""),
                        textvariable=self.second)
        self.secondEntry.place(x=180,y=20)

        self.image_area = Canvas(self.app, width=self.IMG_WIDTH, height=self.IMG_HEIGHT, bg='#C8C8C8')
        self.image_area.pack(pady=(10,0))

        self.show_area = Button(self.app, width=20, text='Done', font='none 12', command=self.show_mask)
        self.show_area.pack(pady=(0,5))

        self.lower = np.array([0, 0, 0])
        self.higher = np.array([255, 255, 255])
       
        # self.save_image = Button(self.app, width=20, text='SAVE IMAGE', font='none 12', command=self.save_image)
        # self.save_image.pack()

    def run(self):
        self.submit()
        self.app.mainloop()

        # self.app.destroy()
        print("Done!")

        return [self.lower, self.higher]

    def blur_img(self, img, factor = 20):
        # kW = int(img.shape[1] / factor)
        # kH = int(img.shape[0] / factor)

        # #ensure the shape of the kernel is odd
        # if kW % 2 == 0: kW = kW - 1
        # if kH % 2 == 0: kH = kH - 1

        # blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
        
        blurred_img = np.zeros([img.shape[0],img.shape[1],3],dtype=np.uint8)
        blurred_img.fill(255)
        return blurred_img


    def takePhoto(self):

        vidcap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        _,self.image = vidcap.read()

        blurred_img = self.blur_img(self.image, factor = 2)

        faces = models.face_cascade.detectMultiScale(self.image, 1.1, 4)
        faces = get_largest_frame(faces)

        blur_margin = 0
        for (x, y, w, h) in faces:
            detected_face = self.image[int(y)-blur_margin:, int(x)-blur_margin:int(x+w)+blur_margin]
            blurred_img[y-blur_margin:, x-blur_margin:x+w+blur_margin] = detected_face
            self.image = blurred_img

        if models.is_frozen:
            cv2.imwrite(os.path.join(models.EXE_LOCATION,'data','reference_color.jpg'), self.image)
        else:
            cv2.imwrite(os.path.join(models.EXE_LOCATION,'data','reference_color.jpg'), self.image)

        b,g,r = cv2.split(self.image)
        self.image = cv2.merge((r,g,b))

        self.image = Image.fromarray(self.image)
        self.image_for_mask_multiplication = self.image

        self.image = ImageTk.PhotoImage(self.image)
        self.image_area.create_image(0, 0, image=self.image, anchor='nw')

        self.select_area()
        
    def get_x_and_y(self,event):
        # global lasx, lasy
        self.lasx, self.lasy = event.x, event.y

    def draw_smth(self, event):
        # global lasx, lasy
        self.image_area.create_line((self.lasx, self.lasy, event.x, event.y), fill='red', width=3)
        self.lasx, self.lasy = event.x, event.y
        
        if self.lasx < 500 and self.lasx >=0 and self.lasy < 400 and self.lasy >=0:
            self.mask[self.lasy][self.lasx] = 0 
            self.mask[self.lasy+1][self.lasx+1] = 0 
            self.mask[self.lasy-1][self.lasx-1] = 0 
            self.mask[self.lasy+1][self.lasx-1] = 0 
            self.mask[self.lasy-1][self.lasx+1] = 0 

    def select_area(self):
        self.image_area.bind("<Button-1>", self.get_x_and_y)
        self.image_area.bind("<B1-Motion>", self.draw_smth)

    def return_shape(self,image_in):
        self.image = image_in
        gray = image_in
        edged = cv2.Canny(gray, 30, 200) 

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        cv2.drawContours(self.image, contours, -1, (0, 0, 0), 3)  
        th, im_th = cv2.threshold(self.image, 200, 255, cv2.THRESH_BINARY_INV)
        im_floodfill = im_th.copy()
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), (255,255,255))
        im_floodfill = np.abs(im_floodfill-np.ones((self.IMG_HEIGHT, self.IMG_WIDTH))*255)
        return im_floodfill
        #cv2.imshow("Floodfilled Image", im_floodfill)


    def show_mask(self):
        # global image_for_mask_multiplication
        # global cropped_image, cropped_image_cv2
        mask_3_channels = np.ones((self.IMG_HEIGHT, self.IMG_WIDTH, 3)) 

        image_mattt = (self.mask * 255).astype(np.uint8)
        the_real_mask = self.return_shape(image_mattt)
        mask_3_channels[:,:,0] = the_real_mask/255
        mask_3_channels[:,:,1] = the_real_mask/255
        mask_3_channels[:,:,2] = the_real_mask/255

        real_area = np.array(self.image_for_mask_multiplication) * mask_3_channels
        real_area = Image.fromarray(np.uint8(real_area)).convert('RGB')
        
        self.cropped_image = real_area.convert("RGB") # RGBA
        datas = self.cropped_image.getdata()

        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        self.cropped_image.putdata(newData)

        self.cropped_image_cv2 = np.array(self.cropped_image) 
        # Convert RGB to BGR 
        self.cropped_image_cv2 = self.cropped_image_cv2[:, :, ::-1].copy()

        self.process()

        self.cropped_image.show()

    def remove_white(self,img):

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

    def process(self):
        # global cropped_image, cropped_image_cv2

        self.cropped_image_cv2 = self.remove_white(self.cropped_image_cv2)
        height, width, channels = self.cropped_image_cv2.shape

        lower = np.array([255,255,255])
        higher = np.array([0,0,0])
        # crop = np.asarray(cropped_image)
        # print(crop)

        for y in range(0, height):
            for x in range(0, width):
                pixel = self.cropped_image_cv2[y,x]
                if (not np.array_equal(pixel, np.array([255,255,255]))
                    and not np.array_equal(pixel, np.array([0,0,0]))):

                    lower[0] = min(pixel[0], lower[0])
                    lower[1] = min(pixel[1], lower[1])
                    lower[2] = min(pixel[2], lower[2])
                    
                    higher[0] = max(pixel[0],higher[0])
                    higher[1] = max(pixel[1],higher[1])
                    higher[2] = max(pixel[2],higher[2])
                    
        print(lower)
        print(higher)
        self.lower = lower
        self.higher = higher

        self.app.destroy()
        # return [lower, higher]


    def submit(self):
        temp = int(self.minute.get())*60 + int(self.second.get())     
        while temp >-1:
            mins,secs = divmod(temp,60)

            self.minute.set("{0:2d}".format(mins))
            self.second.set("{0:2d}".format(secs))   
            self.app.update()
            # time.sleep(1)   
            if (temp == 0):
                self.takePhoto()
                # messagebox.showinfo("Time Countdown", "Time's up ")

            temp -= 1
