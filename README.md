# Traffic-Light-Color-Detection

## Description

This project is a machine learning and computer vision application developed for a university course. The goal is to accurately detect the color of traffic lights in images using a combination of machine learning models and image processing techniques.


## Code Pipeline

![Code Pipeline](https://github.com/ghadyg/Traffic-Light-Color-Detection/blob/main/code-pipeline.png)

The process of detecting the traffic light color involves several key steps:

1. **Image Input**: The program starts by taking an image containing traffic lights. The images used in this project are stored in the `images` directory.

2. **Traffic Light Detection**: We utilize a pre-trained model, `ssd_mobilenet_v2_320x320_coco17_tpu-8`, to detect traffic lights within the image. The model outputs the bounding box coordinates for each detected traffic light.

3. **Bounding Box and Cropping**: Using the bounding box coordinates, the program draws rectangles around each detected traffic light. It then crops these regions to focus on the traffic lights for further analysis.

4. **Dividing the Traffic Light**: Each cropped traffic light is divided into three equal parts vertically:
   - **Green Part**: The top third of the traffic light.
   - **Yellow Part**: The middle third.
   - **Red Part**: The bottom third.

5. **Color Mask Application**: For each part, a color mask is applied:
   - **Green Mask** for the top part.
   - **Yellow Mask** for the middle part.
   - **Red Mask** for the bottom part.

6. **Noise Reduction**: A median filter is applied to each masked part to reduce "salt and pepper" noise, which helps in better contour detection.

7. **Contour Detection**: The program checks for white contours in each part:
   - If a white contour is detected in the green part, the light is green.
   - If a white contour is detected in the yellow part, the light is yellow.
   - If a white contour is detected in the red part, the light is red.

8. **Output Result**: The detected color of each traffic light is then displayed on top of the respective traffic light in the image.


## Results
### For a green Traffic light:
![Green Light](https://github.com/ghadyg/Traffic-Light-Color-Detection/blob/main/results/Picture4.jpg)


### For a red Traffic light:
![Red Light](https://github.com/ghadyg/Traffic-Light-Color-Detection/blob/main/results/Picture5.jpg)


### For a yellow Traffic light:
![Yellow light](https://github.com/ghadyg/Traffic-Light-Color-Detection/blob/main/results/Picture6.jpg)

