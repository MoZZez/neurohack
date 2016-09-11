import cv2
import numpy as np

def frame_selector(Namefil):
    vc = cv2.VideoCapture((Namefil))
    
    count=0.0
    pixcount =0.0
    percent =0.0
    r,f=vc.read()
    avg = np.float32(f)
    counter=0
    maxpercent=0

    while True:
            r,f=vc.read()
            #r,f=vc.read()
          #  print str(r)
            
            
            new=f
            f= cv2.medianBlur(f,9)
            f= cv2.medianBlur(f,9)
           # sobelx =cv2.Sobel(f,cv2.CV_64F,1,1,ksize=5)
           # sobely =cv2.Sobel(f,cv2.CV_64F,0,1,ksize=5)
            laplac = cv2.Laplacian(f,cv2.CV_64F)
            #laplac2 = cv2.blur(laplac,(3))
            #f=f*10
            kernel = np.ones((3,3),np.float32)/25
            dst = cv2.filter2D(laplac,-1,kernel)
            #dst2 = cv2.filter2D(sobelx,-1,kernel)
            dst = dst.astype(np.uint8)
            ret,tocount=cv2.threshold(dst,100,255,cv2.THRESH_BINARY)

            count =0.0
            pixcount=0.0
            
            for ims in tocount:
                for pix in ims:
                  #pix from 0 - 255
                  count=count+1
                  if pix >3:
                      pixcount=pixcount+1
            percent = pixcount/count
            #print "Percent"
            #print percent
            if percent>maxpercent:
                numf=vc.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                maxpercent=percent
            print(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            if vc.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)>=vc.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT ):
                break
            vc.set(1,int(numf))
            r,f=vc.read()
            
    return f
name="se000.avi"
img=frame_selector(name)
cv2.imshow("img",img)
cv2.waitKey()
