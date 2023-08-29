# OCR-Project

This project focuses on online documents consisting of text, tables, and images. 

Image is taken as input. 

Output is a copy of the image with bounding boxes and lines holding any image or table within the image, along with the text from the image.

Steps:
1. **Image Preprocessing:** Input image is preprocessed to enhance table lines and text regions. 
The image goes through a 3 step process:
 Grayscale -> Thresholding - > Dilation

2. **Cell Detection:** Contours are identified to detect individual cells within the table to accurately convert table data to a csv file.

3. **Cropping and Saving:** Each cell's region is cropped from the original image and put into a csv file.
