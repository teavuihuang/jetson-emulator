Jetson AI-Computer Emulator 
==================================
The Jetson Emulator emulates the NVIDIA Jetson AI-Computer's Inference and Utilities API for image classification, object detection and image segmentation (i.e. imageNet, detectNet and segNet). The intended users are makers, learners, developers and students who are curious about AI computers and AI edge-computing but are not ready or able to invest in an actual device such as the NVIDIA Jetson Nano Developer Kit (https://developer.nvidia.com/embedded/jetson-nano-developer-kit). E.g. this allows every student in a computer class to have their own personal AI computer to explore and experiment with. This Jetson Emulator presents a pre-configured, ready-to-run kit with 2 virtual HDMI displays and 4 virtual live-cameras. This enables usage familiarisation with the Jetson API and experimentation with AI computer vision inference. It is a great way to quickly and easily get 'hands-on' with Jetson and experience the power of AI.



System Setup
===================
No setup is required. The Jetson Emulator is virtually pre-configured with JetPack and operates off-line (no Firewall ports need to be configured as video streaming is simulated). There is no need to run the Docker Container nor build the jetson-inference project. The user can start coding their first Jetson AI program as soon as the Python package is installed. The video output is displayed inline on the Jupyter notebook.



Importing the libary
===================
The Jetson Emulator library provides a subset of the NVIDIA "jetson.inference" library API. To import the jetson_emulator module use:
```python
import jetson_emulator.inference as inference
import jetson_emulator.utils as utils
```

This way, the module can referred to as "inference" and "utils" throughout the rest of the application. 
The module can also be imported using the name "jetson_emulator.inference" instead of "jetson.inference" for existing code using the NVIDIA library.
Reference API documentation for the "jetson-inference" Python libraries can be found below.

| Jetson Inference API      | URL                                                                                     | 
|:-------------------------:|:---------------------------------------------------------------------------------------:|
| imageNet/detectNet/segNet | https://rawgit.com/dusty-nv/jetson-inference/dev/docs/html/python/jetson.inference.html |
| jetson-utils              | https://rawgit.com/dusty-nv/jetson-inference/dev/docs/html/python/jetson.utils.html     |


Image Recognition
===================
Image recognition can be performed by classifying images with ImageNet. The imageNet object accepts an input image and outputs the probability for each class. The supported image recognition network is 'GoogleNet' and can recognize up to 1000 different classes of objects from the ImageNet ILSVRC dataset, like different kinds of fruits and vegetables, many different species of animals, along with everyday man-made objects like vehicles, office furniture, sporting equipment, etc.

*Note: This imageNet only works with simulated images, and detects a maximum of 1 object.*


Sample code for imageNet (Image Recognition):
```python
import jetson_emulator.inference as inference
import jetson_emulator.utils as utils

# load the recognition network
net = inference.imageNet("googlenet")
for x in range(1,6):
	# emulator API to generate sample images for imageNet
	filename = net.emulatorGetImageFile()      
	img = utils.loadImage(filename) 
	class_idx, confidence = net.Classify(img)            
	class_desc = net.GetClassDesc(class_idx)            
	print("image "+str(x)+" is recognized as '{:s}' (class #{:d}) with {:f}% confidence".
	format(class_desc, class_idx, confidence*100))
```


Sample output for imageNet:
```
image 1 is recognized as 'dugong, Dugong dugon' (class #149) with 79.249234% confidence
image 2 is recognized as 'gasmask, respirator, gas helmet' (class #570) with 69.749061% confidence
image 3 is recognized as 'cello, violoncello' (class #486) with 48.442260% confidence
image 4 is recognized as 'butternut squash' (class #942) with 43.431145% confidence
image 5 is recognized as 'accordion, piano accordion, squeeze box' (class #401) with 51.372652% confidence
```



Object Detection
===================
Object detection can be performed with DetectNet. It finds where in the frame various objects are located by extracting their bounding boxes. The detectNet object accepts an image as input, and outputs a list of coordinates of the detected bounding boxes along with their classes and confidence values. The supported detection model is a 91-class SSD-Mobilenet-v2 model trained on the MS COCO dataset, which include people, vehicles, animals, and assorted types of household objects. The overlay is fixed to 'boxes'. 

*Note: This detectNet only works with the virtual live-cams, and detects a maximum of 3 objects.*


| Device     | Video source URI                                   | Desc           |
|:----------:|:--------------------------------------------------:|:---------------:
| Camera #1  | rtsp://jetson_emulator:554/detectNet/road_cam/4k   | Moving traffic |
| Camera #2  | rtsp://jetson_emulator:554/detectNet/random_cam/4k | Random objects |


| Device     | Video output URI | Desc           |
|:----------:|:----------------:|:---------------:
| Display #0 |   display://0    | -              |
| Display #1 |   display://1    | 4K             |


Camera #1 features a "real-time" video stream of a moving traffic. From left to right of the video, the detectable objects in the traffic is person, bicycle and car. The view of the car is obstructed approximately 50% of the time, hence it may not always be detectable. Camera #2 is a video stream of 1 to 3 random objects.


Sample code for DetectNet (Object Detection):
```python
import jetson_emulator.inference as inference
import jetson_emulator.utils as utils

network = "ssd-mobilenet-v2"
net = inference.detectNet(network, threshold=0.5)
input_URI = "rtsp://jetson_emulator:554/detectNet/road_cam/4k" 
input  = utils.videoSource(input_URI, argv="")
output = utils.videoOutput("display://1", argv="")
img = input.Capture()
detections = net.Detect(img, "box")
output.SetStatus("{:s} | Network {:.0f} FPS".format(network, net.GetNetworkFPS()))
output.Render(img)
print("detected {:d} objects in image\n".format(len(detections)) )
for detection in detections:
	print("class_desc:", net.GetClassDesc(detection.ClassID))  
	print(detection)
```


Sample output for DetectNet:
```
detected 2 objects in image

class_desc: person
<jetson.inference.Detection object>
   -- ClassID: 2
   -- Confidence: 0.814237
   -- Left: 146.0
   -- Top: 466.0
   -- Right: 204.0
   -- Bottom: 504.0
   -- Width: 58.0
   -- Height: 38.0
   -- Area: 2204.0
   -- Center: (175.0, 485.0)

class_desc: bicycle
<jetson.inference.Detection object>
   -- ClassID: 3
   -- Confidence: 0.966989
   -- Left: 409.0
   -- Top: 451.0
   -- Right: 483.0
   -- Bottom: 519.0
   -- Width: 74.0
   -- Height: 68.0
   -- Area: 5032.0
   -- Center: (446.0, 485.0)
```


Image Segmentation
===================
Semantic segmentation with SegNet is based on image recognition, except the classifications occur at the pixel level as opposed to the entire image. SegNet accepts as input a 2D image, and outputs a second image with the per-pixel classification mask overlay. Each pixel of the mask corresponds to the class of object that was classified. The pre-trained semantic segmentation model supported is the 21-class Pascal VOC 'fcn-resnet18-voc-320x320'. It contains various people, animals, vehicles, and household objects. Visualization is set to 'mask'.

*Note: This segNet only works with the virtual live-cams, and detects a maximum of 2 objects.*


| Device     | Video source URI                                 | Desc           |
|:----------:|:------------------------------------------------:|:---------------:
| Camera #1  | rtsp://jetson_emulator:554/segNet/sofa_cam/4k    | Moving person  |
| Camera #2  | rtsp://jetson_emulator:554/segNet/random_cam/4k  | Random objects |


| Device     | Video output URI | Desc           |
|:----------:|:----------------:|:---------------:
| Display #0 |   display://0    | -              |
| Display #1 |   display://1    | 4K             |


Camera #1 features a "real-time" video stream of a moving person resting on a sofa. The person tosses back and forth, and the sofa contract and expands accordingly. Camera #2 is a video stream of 2 random objects, one on top of the other.


Sample code for SegNet (Image Segmentation):
```python
import jetson_emulator.inference as inference
import jetson_emulator.utils as utils
import numpy as np

network = "fcn-resnet18-voc-320x320"
net = inference.segNet(network, None)
input_URI = "rtsp://jetson_emulator:554/segNet/sofa_cam/4k"
input  = utils.videoSource(input_URI, None)
output = utils.videoOutput("display://1", None)
img = input.Capture()
net.Process(img, ignore_class="void")

# get image mask for video output
img_mask = utils.cudaAllocMapped(width=img.shape[1], height=img.shape[0], format=img.format) 
net.Mask(img_mask, img_mask.shape[1], img_mask.shape[0], filter_mode="linear")
output.SetStatus("{:s} | Network {:.0f} FPS".format(network, net.GetNetworkFPS()))
output.Render(img_mask)

# get class mask to calculate histogram
grid_width, grid_height = net.GetGridSize()
class_mask = utils.cudaAllocMapped(width=grid_width, height=grid_height, format="gray8")
net.Mask(class_mask, grid_width, grid_height, filter_mode="linear")
class_mask_np = utils.cudaToNumpy(class_mask)

# compute the number of times each class occurs in the mask
max_class = np.amax(class_mask_np)
class_histogram, _ = np.histogram(class_mask_np, bins=max_class+1, density=False)
print('-----------------------------------------')
print(' ID  class name         count   %')
print('-----------------------------------------')
for n in range(max_class+1):
	percentage = float(class_histogram[n]) / float(grid_width * grid_height)
	print(' {:>2d}  {:<18s} {:>5d}   {:f}'.format(n, net.GetClassDesc(n), class_histogram[n], percentage)) 
```

Sample output for SegNet:
```
-----------------------------------------
 ID  class name         count   %
-----------------------------------------
  0  background         24000   0.234375
  1  aeroplane              0   0.000000
  2  bicycle                0   0.000000
  3  bird                   0   0.000000
  4  boat                   0   0.000000
  5  bottle                 0   0.000000
  6  bus                    0   0.000000
  7  car                    0   0.000000
  8  cat                    0   0.000000
  9  chair                  0   0.000000
 10  cow                    0   0.000000
 11  diningtable            0   0.000000
 12  dog                    0   0.000000
 13  horse                  0   0.000000
 14  motorbike              0   0.000000
 15  person             53120   0.518750
 16  pottedplant            0   0.000000
 17  sheep                  0   0.000000
 18  sofa               25280   0.246875
```

Author and Citation
===================
Tea Vui Huang. (2020, October 19). 
Jetson AI-Computer Emulator. http://doi.org/10.5281/zenodo.4106061
