import pandas as pd
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from google.cloud import vision
from google.oauth2 import service_account
import io

credentials_path = 'api-key.json'
credentials = service_account.Credentials.from_service_account_file(credentials_path)
client = vision.ImageAnnotatorClient(credentials=credentials)


    
class OCR:

    def __init__(self, image_path):

        self.image_path = image_path
        self.cropped = None
        self.bounding_boxes = []
        self.rows = []
        self.dataframe = None

    def ocr(image):
        image = vision.Image(content=image)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            text = texts[0].description
            text_with_newlines = text.replace("\n", " ")
            return text_with_newlines
        else:
            return "No text found in the image."

    

    def process_image(self, image_path):
        # pre processing image
        img = cv2.imread(image_path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # set bounding box color
        color = (255, 0, 0)

        # contour and line detection in image

        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        edges = cv2.Canny(gray_image, 100, 200, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=5)

        # Find coordinates of table

        min_x1 = float('inf')
        min_y1 = float('inf')
        max_x2 = float('-inf')
        max_y2 = float('-inf')

        for line in lines:
            for x1, y1, x2, y2 in line:
                min_x1 = min(min_x1, x1)
                min_y1 = min(min_y1, y1)
                max_x2 = max(max_x2, x2)
                max_y2 = max(max_y2, y2)

        # Cropping table

        image_pil = Image.fromarray(img)
        box = (min_x1, min_y1, max_x2, max_y2)
        self.cropped = image_pil.crop(box)

        # Drawing bounding boxes
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 40 and h > 40:
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)

        # draw table lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 4)

        # Extract text from image
        detected_text = ocr(cv2.imencode('.jpg', img)[1].tobytes())
        print(detected_text)
        # Display image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        
        return self.cropped

    def draw_bounding_boxes(self, cropped):
        # convert image back to np array
        self.cropped = np.array(cropped)

        # Calculate padding as 10% of the image height and add it to all sides of the image
        image_height = self.cropped.shape[0]
        padding = int(image_height * 0.1)
        padded_img = cv2.copyMakeBorder(cropped, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # converting to grayscale and binary image
        gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # make text regions white and background black
        inverted_image = cv2.bitwise_not(thresh_image)

        # remove vertical lines
        hor = np.array([[1,1,1,1,1,1]])
        vertical_eroded_img = cv2.erode(inverted_image, hor, iterations=10)
        vertical_eroded_img = cv2.dilate(vertical_eroded_img, hor, iterations=10)

        # remove horizontal lines
        ver = np.array([[1], [1], [1], [1], [1], [1]])
        horizontal_eroded_img = cv2.erode(inverted_image, ver, iterations=10)
        horizontal_eroded_img = cv2.dilate(horizontal_eroded_img, ver, iterations=10)

        # Combine horizontally and vertically cleaned images
        combined_image = cv2.add(vertical_eroded_img, horizontal_eroded_img)

        # dilate image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        combined_image_dilated = cv2.dilate(combined_image, kernel, iterations=5)

        # Subtract dilated image from the inverted image to remove lines
        image_without_lines = cv2.subtract(inverted_image, combined_image_dilated)

        # noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image_without_lines_noise_removed = cv2.erode(image_without_lines, kernel, iterations=1)
        image_without_lines_noise_removed = cv2.dilate(image_without_lines_noise_removed, kernel, iterations=1)

        # define kernel to remove gaps between words in cells
        kernel_to_remove_gaps_between_words = np.array([
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1]
            ])
        dilated_image = cv2.dilate(thresh_image, kernel_to_remove_gaps_between_words, iterations=5)
        kernel = np.ones((5,5), np.uint8)
        # convert text blocks to blobs
        dilated_image = cv2.dilate(dilated_image, kernel, iterations=2)

        # find and draw contours
        result = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = result[0]
        image_with_contours_drawn = cropped.copy()
        cv2.drawContours(image_with_contours_drawn, contours, -1, (0, 0, 255), 3)

        # draw bounding boxes
        bounding_boxes = []
        image_with_all_bounding_boxes = cropped.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # check for very small boxes that are not needed
            if h > 0.01*self.cropped.shape[0] and w > 0.01*self.cropped.shape[1]:
                bounding_boxes.append((x, y, w, h))
            image_with_all_bounding_boxes = cv2.rectangle(image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)
        plt.imshow(image_with_all_bounding_boxes)
        return bounding_boxes
    
    def find_rows(self, bounding_boxes):
        # Sorting The Bounding Boxes By X And Y Coordinates To Make Rows And Columns

        # finding mean of bounding box heights
        heights = []
        for box in bounding_boxes:
            x, y, w, h = box
            heights.append(h)
        mean =  np.mean(heights)

        # sort boxes by y value
        bounding_boxes.sort(key=lambda x: x[1])

        # group bounding boxes with similar y coordinates
        rows = []
        half_mean = mean / 2
        curr_row = [bounding_boxes[0]]
        for box in bounding_boxes[1:]:
            curr_box_y = box[1]
            prev_box_y = curr_row[-1][1]
            dist = abs(curr_box_y - prev_box_y)
            if dist <= half_mean:
                curr_row.append(box)
            else:
                self.rows.append(curr_row)
                curr_row = [box]


        # sort rows by x value
        for row in self.rows:
            row.sort(key=lambda x: x[0])

        return self.rows

    def get_dataframe_and_csv(self, rows):
        table = []
        current_row = []
        cells = []
        # detect text from each cell in row and all rows to table
        for row in self.rows:
            for box in row:
                x, y, w, h = box
                y = y-5
                cropped_im = self.cropped[y:y+h, x:x+w]
                cells.append(cropped_im)
                detected_text = ocr(cv2.imencode('.jpg', cropped_im)[1].tobytes())
                current_row.append(detected_text)
            table.append(current_row)
            current_row = []

        detected_text = detected_text.strip()
        # convert table array to dataframe
        df = pd.DataFrame(table)
        csv = df.to_csv()
        return df, csv

 


