# Image Color Identification

## Overview
In this repo, I applied the basics of OpenCV, extracted colors from images using KMeans algorithm and filtered images from a collection of images based on RGB values of colors. All of the images were taken from Unsplash.

## Import Libraries
```bash
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

%matplotlib inline
```
Import the necessary libraries including matplotlib.pyplot, numpy , Counter to extract the count and cv2 for OpenCV. KMeans algorithm is part of the sklearn's cluster subpackage. To compare colors we first convert them to lab using rgb2lab and then calculate similarity using deltaE_cie76. Finally,import os to combine paths while reading files from a directory.

## Testing on image with OpenCV
```bash
image = cv2.imread('/content/sampleimg.jpeg')
print('The input type is {}'.format(type(image)))
print("Shape: {}".format(image.shape))
plt.imshow(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')
resized_image = cv2.resize(image, (550,550))
plt.imshow(resized_image)
```
I tried some processing on images with OpenCV such as reading image, converting colors and resizing image.

![image](https://user-images.githubusercontent.com/87477460/160425477-3f8d7b9f-7e5f-4b34-81c2-ded99546b7e4.png)

## Color Identification
```bash
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
```
The pie chart is going to output the colors as hex values so we firstly need to define a function that change rgb values to hex values.

```bash
def get_image(image_path):
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  return image
```
Supply the image path as an argument.Read the image and get in rgb space.

```bash
def get_colors(image, number_of_colors,show_chart):
  modified_image = cv2.resize(image,(550,550),interpolation = cv2.INTER_AREA)
  modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1],3)

  clf = KMeans(n_clusters = number_of_colors)
  labels = clf.fit_predict(modified_image)

  counts = Counter(labels)
  counts = dict(sorted(counts.items()))

  center_colors = clf.cluster_centers_
  ordered_colors = [center_colors[i] for i in counts.keys()]
  hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
  rgb_colors = [ordered_colors[i] for i in counts.keys()]

  if(show_chart):
    plt.figure(figsize = (8,6))
    plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

  return rgb_colors
```
This part is the complete code that read the image, extract the major colors and show as a pie chart. It takes three arguments. The image we want to use, the number of colors we want to extract and a boolean that the user wants to show the pie chart or not. Firstly, the image is resized into 550x550. This step is not necessary but I applied since it can reduce the number of pixels and time needed.

KMeans expects the input to be of two dimensions, so I use Numpy’s reshape function to reshape the image data.KMeans algorithm creates clusters which will be our top colors and fit. The results are kept in labels variable. 

Counter is used to get count of all labels.To find the colors, I use clf.cluster_centers_. The center_colors are iterated into ordered_colors and again ordered_colors is changed into hex_colors and rgb_colors.

```bash
get_colors(get_image('/content/sampleimg.jpeg'), 10, True)
```
If show_chart is True, it plot a pie chart with each pie chart portion defined using count.values(), labels as hex_colors and colors as ordered_colors. 

![image](https://user-images.githubusercontent.com/87477460/160420464-e458cd93-0ce7-4b55-9212-d4acb0ce112e.png))

## Search Images Using Colors
```bash
IMAGE_DIRECTORY = 'images'
COLORS = {
    'PURPLE': [170,116,211],
    'ORANGE': [250,83,0],
    'BLUE': [0, 0, 128]
}
images = []

for file in os.listdir(IMAGE_DIRECTORY):
  if not file.startswith('.'):
    images.append(get_image(os.path.join(IMAGE_DIRECTORY,file)))
```
In this case, I supply three colors' RGB values (Purple, Orange and Blue) and the system is going to filter the images based on this three colors.

```bash
plt.figure(figsize=(20,10))
for i in range (len(images)):
  plt.subplot(1,len(images),i+1)
  plt.imshow(images[i])
```
This is how I show the images form my image directory.

![image](https://user-images.githubusercontent.com/87477460/160432623-7aa1dbb3-ba06-415f-9fad-bc95ce75dc40.png)

## Match Images By Color
```bash
def match_image_by_color(image, color, threshold = 60, number_of_colors = 10): 
    
    image_colors = get_colors(image, number_of_colors, False)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

    select_image = False
    for i in range(number_of_colors):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        if (diff < threshold):
            select_image = True
    
    return select_image
```
Previously defined method get_colors extract the image colors using RGB format. I use the method rgb2lab to convert the selected color to a format we can compare. The forl oop iterates over all the colors retrieved from the image.

For each color, the loop changes it to lab, finds the delta (difference) between the selected color and the color in iteration. If the delta is less than the threshold, the image is selected as a match with the color. We need the delta-threadhold comparison because the image has many shades and our selected colors cannot always match with the color in images.

The threshold defines how different can the colors of the image and selected color be.

```bash
def show_selected_images(images, color, threshold, colors_to_match):
    index = 1
    
    for i in range(len(images)):
        selected = match_image_by_color(images[i],color,threshold,colors_to_match)
        
        if (selected):
            plt.subplot(1, 5, index)
            plt.imshow(images[i])
            index += 1
```
This function iterates over all images, calls the above function to filter them based on color and displays them on the screen using imshow.

## Search for Purple
```bash
plt.figure(figsize = (20,10))
show_selected_images(images, COLORS['PURPLE'],60,5)
```
![image](https://user-images.githubusercontent.com/87477460/160434876-44e3ae7f-23e8-4ecc-ba66-00c110bc5295.png)

## Search for Orange
```bash
plt.figure(figsize = (20,10))
show_selected_images(images, COLORS['ORANGE'],60,5)
```
![image](https://user-images.githubusercontent.com/87477460/160434987-0da4ce3b-8f4d-47ea-a306-81ea045c855d.png)

## Search for Blue
```bash
plt.figure(figsize = (20,10))
show_selected_images(images, COLORS['BLUE'],60,5)
```
![image](https://user-images.githubusercontent.com/87477460/160435119-1f0ffba9-c40f-447a-af01-f26cf072871a.png)
