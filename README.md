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
    The book cover image, which I took a picture of it, comes from school's library. 
    The text on the book cover has different color and size, capital letters and the caligraphy letters. 
    While the driver license image comes from internet, contains multiple styles of letters.
### functions and algorithms
    The image preprocessing part used varies functions from the OPENCV library, for example
    cv2.cvtColor(), cv2.threshold(),cv2.GaussianBlur(),cv2.Canny(),cv2.findContours(),cv2.warpPerspective() and so on
    For image displaying I use cv.imshow() and matpolitlib. 
    In version 2 & 3 I add Trackbar and mouse call-back fucntion into the program.
### result
    
    
