
############ definition and import library #################

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt  
import pytesseract #OCR TOOL, RGB form(opencv-BGR so need to transfer img first)
import time

start_time = time.time()

# import Tesseract.exe path
pytesseract.pytesseract.tesseract_cmd = r'E:\tesseract-OCR\tesseract.exe'
# load img
img_path = "G:\\Master\\CDS540_CV\\Ind_assgin\\Test Sample\\book.jpg"
IMG = cv.imread(img_path)

############### subfunction definition ###############
# resize
def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[0:2] # get h and w from img data
    # original
    if width is None and height is None:
        return image
    # specific height
    if width is None:
        r = height / float(h) # get the ratio
        dim = (int(w * r), height)
    # specific width
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv.resize(image, dim, interpolation=inter)
    return resized

# warp_find 4 points
def order_points(pts):
    # define a matrix
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# warp_transfer
def four_point_transform(image, pts):
    # order the 4 points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # get the max distance between points （tl -br）
    widthA = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    widthB = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    max_width = max(int(widthA), int(widthB))
    heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    max_height = max(int(heightA), int(heightB))
    # new location points
    dst = np.array([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]], dtype='float32')
    # matrix for warp
    M = cv.getPerspectiveTransform(rect, dst)
    # warp
    warped = cv.warpPerspective(image, M, (max_width, max_height))
    return warped



# display_plt
def img_display(titles,images,n):
    plt.figure(figsize=(16,8))

    for i in range(n):
        plt.subplot(2,4,i+1),plt.imshow(cv.cvtColor(images[i],cv.COLOR_BGR2RGB),extent=[0, 30, 0, 36])
        plt.title(titles[i])
        plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

#display_opencv
def opencv_display(winname, image):
    cv.imshow(winname, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


#================= Pre-processing =======================#

def Img_preprocessing(image):
    # 2) convert img to grayscale
    Img_Gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    # 3) Threshold
    thresh,Img_Thresh=cv.threshold(Img_Gray,120,255,cv.THRESH_BINARY)
    # 4) guassian blur
    Img_Blurred = cv.GaussianBlur(Img_Thresh,(5,5),0)
    # 5) find edge (Canny) (need to improve this part inorder to get better edge)
    Img_Edge = cv.Canny(Img_Blurred, 70, 200)
    # 6) find and draw all contours
    Contours =  cv.findContours(Img_Edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    Img_Contour_ALL = cv.drawContours(image.copy(), Contours, -1, (0, 0, 255), 1)
    
    # 7) find and draw the largest contour
    # sort contours by area from large to small
    Contours = sorted(Contours, key=cv.contourArea, reverse=True)
    # iterate to find the closest contour 
    for c in Contours:
        # calculate perimeter
        perimeter = cv.arcLength(c, True)
        # get the closest rectangle contour approximatly 
        approx = cv.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    # draw the contour
    Img_Contour_MAX = cv.drawContours(image.copy(), [screenCnt], -1, (0, 0, 255), 1)

    # 8) warp 
    Img_warped = four_point_transform(image_copy, screenCnt.reshape(4, 2) * ratio)
    Scan = cv.cvtColor(Img_warped, cv.COLOR_BGR2GRAY)
    Scan_Thresh = cv.threshold(Scan, 190, 255, cv.THRESH_BINARY)[1]
    Scan_Blurred = cv.GaussianBlur(Scan_Thresh,(3,3),0)

    return image,Img_Gray,Img_Thresh,Img_Edge,Img_Contour_ALL,Img_Contour_MAX,Img_warped,Scan_Blurred


# 9) do OCR
def OCR(Scanned_File):
    # read and print text
    text = pytesseract.image_to_string(Scanned_File)
    print(text)
    # save the ocr result to txt file
    with open('ocr_result.txt', 'w') as f:
        print(text)
        f.write(str(text))
    
    # draw box
    scanned_file_copy = scanned_file.copy()
    hImg, wImg, _ = Scanned_File.shape
    # get box info 
    boxes = pytesseract.image_to_data(Scanned_File)
    print(boxes)
    # split the boxes by rows and enumerate every element in boxes
    for x,b in enumerate(boxes.splitlines()):
        # get rid of the first row(title)      
        if x!=0:
            b = b.split()
            print(b)
            # if OCR the text successfully we will get 12 rows data
            if len(b)==12: 
                # we only use the row 7,8,9,10,12
                x1, y1, x2, y2 = int(b[6]), int(b[7]), int(b[8]), int(b[9]) # get the boxes data
                # draw boxes
                cv.rectangle(Scanned_File, (x1,y1), (x1+x2, y1+y2), (0, 0, 255), 3)
                #put the OCR text(b[11]) on the image boxs, location(x1,y1)
                cv.putText(Scanned_File, b[11], (x1,y1), cv.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
    


    #display result, YOU CAN annotate line 166 to line 175 to get the OCR result txt file 
    images_ocr = scanned_file_copy,Scanned_File
    titles_ocr = "scanned file","ocr result"
    plt.figure(figsize=(16,8))

    for i in range(2):            
        plt.subplot(1,2,i+1),plt.imshow(cv.cvtColor(images_ocr[i],cv.COLOR_BGR2RGB))
        plt.title(titles_ocr[i])
        plt.axis('off')
    plt.tight_layout()  # auto adjust the distance between subplot
    plt.show()
    

############### Main function ###############
ratio = IMG.shape[0] / 500.0
image_copy = IMG.copy()
# 1) resize
Img_Resized = resize(image_copy, height = 500)
# 2) preprocess
Img_result = Img_preprocessing(Img_Resized)
# 3) display final image
img_titles = ["original img","gray img","thresh img","Img_edge","Img_Contour_ALL","Img_Contour_MAX","Img_warped","Scan_Thresh"]
# make line 188 green, and you can run the program faster to get the OCR result
img_display(img_titles,Img_result,8)
# 4) save scanned file
cv.imwrite('./scanned_file.jpg', Img_result[-1])
# 5) OCR scanned file
scanned_file = cv.imread("./scanned_file.jpg")
OCR(scanned_file)

with open("ocr_result.txt", "r") as file:
    print(file.read())

#####################################################################




end_time = time.time()

total_time = end_time - start_time
print("Total running time: {:.2f} seconds".format(total_time))


