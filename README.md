# CDS540_IndividualAssignment_1
## Assignment requirement
    Image preprocession and use OCR tool to detect the text on the image, draw boxes around the text
## Environment 
    OPENCV 3
    Python 3.7
    Pytesseract (OCR TOOL)
    VSCODE
## Brief introduction of the program
### sample images
    The test image I used is "book.jpg" (version 1 & 2) and "dri_lic.png" (version 3). 
    The first image is a book cover which I think is very suitable for this project. 
    The book cover image, which is the picture I took from school's library. 
    The text on the book cover has different color and size, capital letters and the caligraphy letters, they come in different thickness too. 
    While the driver license image comes from internet, contains multiple styles of letters on a very messy and colorful background.
### functions and algorithms
    The image preprocessing part used varies functions from the OPENCV library, for example
    cv2.cvtColor(), cv2.threshold(),cv2.GaussianBlur(),cv2.Canny(),cv2.findContours(),cv2.warpPerspective() and so on
    For image displaying I use cv.imshow() and matpolitlib. 
    In version 2 & 3 I add Trackbar and mouse call-back fucntion into the program.
### flow chart of Version 1:
    ![image]([https://github.com/user-attachments/assets/d0fe6126-02bd-4c76-8945-33c7d73f9edd](https://github.com/XZJLL/CV_IndividualAssignment/blob/main/CDS540_A1_V1.drawio.png))

### result
    In version 1, the speed of the program is below 1 second and the program gets most of the words and draw boxes around them.
    The result of preprocessed image displayed in windows and the OCR output save in text file and can be printed on terminal. 
    In version 2, users can use Trackbar to adjust the variables in the function, and the result of version 2 is similar to that of version 1.
    In version 3, users can use mouse to adjust the persbective transformation, and select the ROI. The result of version 3 is not as good as the previous two
        because of the resizes and distortion make the image lost many pixels.
### what can be improved
    Maybe use QT to let users choose different combinations of function as they like would be better.
    
