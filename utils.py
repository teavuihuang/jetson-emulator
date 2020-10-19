#
# Copyright (c) 2020, Tea Vui Huang. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random, logging, time
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.ERROR)


def loadImage(opt_filename):
    return opt_filename
    
    
def cudaToNumpy(img_mask):
    # This function is supported only for segNet
    if "segNet" not in img_mask.cudaMemory:
        return np.full([320*320], 0)
        
    # rect_data = 0=ClassID, 1=Left, 2=Right, 3=Top, 4=Bottom
    rect_data = emulatorGetImgData(img_mask.cudaMemory) 
    
    # scale to 320x320
    if (rect_data[3]<320):        
        scale_factor = 1.7 # From 960×544
    else:
        scale_factor = 12  # From 3840×2160 (4K)
    sy1 = int(rect_data[3]/scale_factor)
    sy2 = int(rect_data[4]/scale_factor)
    sy3 = int(rect_data[8]/scale_factor)
    sy4 = 320 # always the bottom     
        
    # grid size: 320x320
    line_background_np  = np.full([320], 0)             #  0 = background
    line_diningtable_np = np.full([320], rect_data[0])  # 15 = person  
    line_sofa_np        = np.full([320], rect_data[5])  # 18 = sofa
    
    # create the rows for individual masks
    block_background_np  = np.repeat(line_background_np , sy1    , axis=0)
    block_diningtable_np = np.repeat(line_diningtable_np, sy2-sy1, axis=0)
    block_sofa_np        = np.repeat(line_sofa_np       , sy4-sy3, axis=0)
    
    # combine the masks
    img_mask_np = np.concatenate((block_background_np, block_diningtable_np, block_sofa_np), axis=0)
    return img_mask_np
        
    
# Allocate CUDA ZeroCopy mapped memory
def cudaAllocMapped(width, height, format):
    if format=="gray8":
        # getting the class_mask
        logging.debug("cudaAllocMapped: getting the class_mask")
    return cudaImage("segNet,blank,00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.jpg")


# Overlay the input image onto the composite output image at position (x,y), not supported  
def cudaOverlay(input_image, output_image, x, y):
    return 0 # cudaSuccess = 0


# Wait for compute device to finish, not supported   
def cudaDeviceSynchronize():    
    return 0 # cudaSuccess = 0


# Custom Emulator API, get image resolution from URI videoSource
def emulatorGetResolutionFromVideoSourceUri(uri):
    # e.g. 3840×2160, 960×544, 480×272, 300×300
    if ("/4k" in uri) or ("display://1" in uri):
        res_width  = 3840
        res_height = 2160
        res_margin = 200
    else:
        res_width  = 960
        res_height = 544
        res_margin = 50
    return res_width, res_height, res_margin


# Custom Emulator API, get image detectNet/segNet data from string stored in cudaMemory
def emulatorGetImgData(img_name):
    # e.g. img_name = "detectNet,random,02,86,443,2467,312,1603,6,1341,3011,640,1817,.jpg"
    rect_data = []
    if type(img_name)!=str:
        return rect_data
    img_data = img_name.split(",")
    if (img_data[0]=="detectNet" or img_data[0]=="segNet") and len(img_data)>=8:
        img_data.pop(0) # remove "detectNet/segNet"
        img_data.pop(0) # remove "(filename)"
        img_data.pop(0) # remove "(counter)"
        img_data.pop()  # remove ".jpg"
        if (len(img_data)%5==0):
            rect_data = list(map(int, img_data))
    return rect_data
    
    

# CUDA image class, emulator stores virtual image data as a string in cudaMemory
class cudaImage:
    def __init__(self, cudaMemory):
        self.cudaMemory = cudaMemory
        
    @property
    def width(self):
        return 320
        
    # Image dimensions in (height, width, channels) tuple, hardcoded
    @property
    def shape(self):
        return (320, 320, 1)
        
    # Pixel format of the image
    @property
    def format(self):
        return "IMAGE_UNKNOWN"



# Virtual cameras
class videoSource:


    def __init__(self, uri, argv):
        self.uri = uri
        logging.debug("videoSource: uri='" + self.uri + "'")
        
        
    def IsStreaming(self):
        return True
        
        
    # Generate a random (x,y) coordinate
    def emulatorGetRandomXY(self, res_width, res_height, res_margin):
        r_left   = random.randint(0, (res_width/2)-res_margin) 
        r_right  = random.randint((res_width/2)+res_margin, res_width) 
        r_top    = random.randint(0, (res_height/2)-res_margin) 
        r_bottom = random.randint((res_height/2)+res_margin, res_height) 
        return r_left, r_right, r_top, r_bottom     
        
        
    # Generate a random jitter
    def emulatorGetJitterXY(self, res_height):
        ds_jitter1 = int(res_height*0.004630)
        ds_jitter2 = int(res_height*0.006944)
        ds_jitter3 = int(res_height*0.009259)
        xs_jitter1 = random.randint(-ds_jitter1, ds_jitter1)
        xs_jitter2 = random.randint(-ds_jitter2, ds_jitter2)
        xs_jitter3 = random.randint(-ds_jitter3, ds_jitter3)
        return xs_jitter1, xs_jitter2, xs_jitter3        
               

    # Capture an image from the video stream
    def Capture(self):
        logging.debug("Capture: videoSource uri='" + self.uri + "'")  
        res_width, res_height, res_margin = emulatorGetResolutionFromVideoSourceUri(self.uri)
        img_name = "detectNet,blank,00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.jpg"
        
        # videoSource for segNet 
        if "/segNet/" in self.uri:
            img_name = "segNet,blank,00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.jpg"
            
            # Choice of 2 video streams
            if ("/sofa_cam" in self.uri) or ("/random_cam" in self.uri):            
                if "/sofa_cam" in self.uri:
                    # segNet: Person tossing around on sofa
                    o1 = 15 # 15 = person 
                    o2 = 18 # 18 = sofa
                else:
                    # segNet: Random objects, one on top of another
                    o1 = random.randint(1, 21) 
                    o2 = random.randint(1, 21)    
                r1 = random.randint(1, int(res_height*0.20))
                r2 = int(r1/2)
                y1 = int((res_height*0.33)) - r2 
                y2 = int((res_height*0.66)) + r2 
                img_name = "segNet,room,02," \
                +str(o1)+",0,"+str(res_width)+","+str(y1)+","+str(y2)+"," \
                +str(o2)+",0,"+str(res_width)+","+str(y2)+","+str(res_height)+",.jpg"     
        
        # videoSource for detectNet 
        elif "/detectNet/" in self.uri:
            img_name = "detectNet,blank,00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.jpg"

            if "/random_cam" in self.uri:
                # detectNet: Random objects, 1 to 3
                img_name = "detectNet,random_cam,00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.jpg"
                num0 = random.randint(1, 3)  # how many objects
                num1 = random.randint(2, 91) # ClassID
                num2 = random.randint(2, 91) # ClassID
                num3 = random.randint(2, 91) # ClassID
            
                r_left1, r_right1, r_top1, r_bottom1 = self.emulatorGetRandomXY(res_width, res_height, res_margin)
                r_left2, r_right2, r_top2, r_bottom2 = self.emulatorGetRandomXY(res_width, res_height, res_margin)
                r_left3, r_right3, r_top3, r_bottom3 = self.emulatorGetRandomXY(res_width, res_height, res_margin)
            
                # rect_data = 0=ClassID, 1=Left, 2=Right, 3=Top, 4=Bottom
                if num0==1:
                    img_name="detectNet,random,0"+str(num0)+"," \
                    +str(num1)+","+str(r_left1)+","+str(r_right1)+","+str(r_top1)+","+str(r_bottom1)+",.jpg"
 
                if num0==2:
                    img_name="detectNet,random,0"+str(num0)+"," \
                    +str(num1)+","+str(r_left1)+","+str(r_right1)+","+str(r_top1)+","+str(r_bottom1)+"," \
                    +str(num2)+","+str(r_left2)+","+str(r_right2)+","+str(r_top2)+","+str(r_bottom2)+",.jpg"

                if num0==3:
                    img_name="detectNet,random,0"+str(num0)+"," \
                    +str(num1)+","+str(r_left1)+","+str(r_right1)+","+str(r_top1)+","+str(r_bottom1)+"," \
                    +str(num2)+","+str(r_left2)+","+str(r_right2)+","+str(r_top2)+","+str(r_bottom2)+"," \
                    +str(num3)+","+str(r_left3)+","+str(r_right3)+","+str(r_top3)+","+str(r_bottom3)+",.jpg"

            elif "/road_cam" in self.uri:
                # detectNet: Road traffic 
                img_name = "detectNet,road_cam,00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.jpg"
                num0 = random.randint(2, 3) # how many objects
                num1 = 2 # 2 = person
                num2 = 3 # 3 = bicycle
                num3 = 4 # 4 = car
            
                # Horizontal jitter for each object
                d_jitter1 = int(res_width* 0.03906)
                d_jitter2 = int(res_width* 0.03646)
                d_jitter3 = int(res_width* 0.03125)            
                x_jitter1 = random.randint(-d_jitter1, d_jitter1) # person
                x_jitter2 = random.randint(-d_jitter2, d_jitter2) # bicycle
                x_jitter3 = random.randint(-d_jitter3, d_jitter3) # car
                        
                # Create 3 horizontal lanes
                x_center1 = x_jitter1 + int(res_width* 0.16) # person
                x_center2 = x_jitter2 + int(res_width* 0.50) # bicycle
                x_center3 = x_jitter3 + int(res_width* 0.83) # car

                # Vertical movement based on 1 decisecond (ds) = 0.1 sec
                decisecond = (datetime.now().microsecond/1000000)
                y_max = int(res_height* 0.90)
                y_jitter1 = y_max-int((decisecond*y_max)) # person
                y_jitter2 = y_max-int((decisecond*y_max)) # bicycle
                y_jitter3 = y_max-int((decisecond*y_max)) # car
                        
                # Give a margin space for the top title
                y_center1 = y_jitter1 + res_margin # jitterish vertical postion of person, with margin
                y_center2 = y_jitter2 + res_margin # jitterish vertical postion of bicycle, with margin
                y_center3 = y_jitter3 + res_margin # jitterish vertical postion of car, with margin
            
                # Apply horizontal scaling jitter
                xs_jitter1, xs_jitter2, xs_jitter3 = self.emulatorGetJitterXY(res_width)
                x_delta1 = xs_jitter1 + int(res_width*0.0260417) # jitterish half-width of person 
                x_delta2 = xs_jitter2 + int(res_width*0.0390625) # jitterish half-width of bicycle
                x_delta3 = xs_jitter3 + int(res_width*0.0520833) # jitterish half-width of car
            
                # Apply vertical jitter
                ys_jitter1, ys_jitter2, ys_jitter3 = self.emulatorGetJitterXY(res_height)
                y_delta1 = ys_jitter1 + int(x_delta1*1.50/2) # jitterish half-height of person
                y_delta2 = ys_jitter2 + int(x_delta2*1.96/2) # jitterish half-height of bicycle
                y_delta3 = ys_jitter3 + int(x_delta3*1.90/2) # jitterish half-height of car
                        
                # rect_data = 0=ClassID, 1=Left, 2=Right, 3=Top, 4=Bottom
                if num0==2:
                    img_name="detectNet,road,0"+str(num0)+"," \
                    +str(num1)+","+str(x_center1-x_delta1)+","+str(x_center1+x_delta1)+","+str(y_center1-y_delta1)+","+str(y_center1+y_delta1)+"," \
                    +str(num2)+","+str(x_center2-x_delta2)+","+str(x_center2+x_delta2)+","+str(y_center2-y_delta2)+","+str(y_center2+y_delta2)+",.jpg"
                
                elif num0==3:
                    img_name="detectNet,road,0"+str(num0)+"," \
                    +str(num1)+","+str(x_center1-x_delta1)+","+str(x_center1+x_delta1)+","+str(y_center1-y_delta1)+","+str(y_center1+y_delta1)+"," \
                    +str(num2)+","+str(x_center2-x_delta2)+","+str(x_center2+x_delta2)+","+str(y_center2-y_delta2)+","+str(y_center2+y_delta2)+"," \
                    +str(num3)+","+str(x_center3-x_delta3)+","+str(x_center3+x_delta3)+","+str(y_center3-y_delta3)+","+str(y_center3+y_delta3)+",.jpg"
        
        return cudaImage(img_name)



# Virtual displays
class videoOutput:


    def __init__(self, uri, argv):
        self.uri = uri
        self.title_bar_status = ""
        logging.debug("videoOutput: uri='" + self.uri + "'")
        
        
    def IsStreaming(self):
        return True
        
        
    def SetStatus(self, title_bar_status):
        self.title_bar_status = title_bar_status
        logging.debug("SetStatus: '" + title_bar_status + "'")


    def Render(self, img):
        logging.debug("videoOutput: uri='" + self.uri + "'")
        if type(img)==cudaImage:
            img_name = img.cudaMemory
        else:
            img_name = "detectNet,blank,00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.jpg"
        logging.debug("Render: img_name='" + img_name + "'")
        rect_data = emulatorGetImgData(img_name)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_facecolor('lightgrey')
        res_width, res_height, res_margin = emulatorGetResolutionFromVideoSourceUri(self.uri)
        ax.axis([0, res_width, 0, res_height]) 
        ax.invert_yaxis()
        ax.set_title(self.title_bar_status)
        ax.axis('off')
        
        # Create a Rectangle patch and add the patch to the Axes
        # rect_data = 0=ClassID, 1=Left, 2=Right, 3=Top, 4=Bottom
        facecolors = ['mediumpurple', 'pink', 'yellow']
        for x in range(0,len(rect_data),5):
            c = int(x/5)
            rx1 = rect_data[x+1]
            rx2 = rect_data[x+2]
            ry1 = rect_data[x+3]
            ry2 = rect_data[x+4]
            width = rx2-rx1
            height = ry2-ry1
            
            if ("segNet" in img_name) and x==0:
                ellipse = patches.Ellipse((width/2,ry1+height/2), width, height,
                edgecolor=facecolors[c],facecolor=facecolors[c],alpha=0.45,angle=0)
                ax.add_patch(ellipse)
            else:
                rect = patches.Rectangle((rx1,ry1),width,height,linewidth=1,
                edgecolor=facecolors[c],facecolor=facecolors[c],alpha=0.45,fill=True)
                ax.add_patch(rect)
            
        plt.show()        
        