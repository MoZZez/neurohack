import cv2

import numpy as np
import os.path


print "skolko v papke"

totalvid=int(input())

print "skolko bolnyh papok(idut snachala)"

kolbol=int(input())
#^ can be set 11 both.


NameStd="se00"
##vc = cv2.VideoCapture("se000.avi")
##r,f=vc.read()
##cv2.imshow("Karl",f)
##cv2.waitKey(1000)
##print "s"
kolpapok=20
papki={1:"Gusarov_Sergey_Anatolievich",2:"Guseinov_Igor_Guseinovich",3:"Hamutova_Ludmila_G",4:"Hurumov_Andrey_Yur",5:"ILIUSHENKO_Vitaliy_Evgenievich",6:"Kuzin_Mihail_Alekseevich",7:"Listikova_Al_bina_Nikolaevna",8:"Lobanov_Pavel_Ivanovich",9:"MAZAEV__L_V",10:"Mazaev_Leonid_Viktorovich",11:"Zemskikh_Aleksandr_Vasil_evich",12:"BOCHAROV_G_A Boldaev_Aleksey_Andreevich",13:"Boldychev_S_M",14:"Bondarenko_Margarita_Pavlovna",15:"Bondarev_Mihail_I",16:"BORISOV_Y_M",17:"KULINCOV_IGOR_VAS",18:"Kuvshinov_Sergei_Ivanovich",19:"KUZNETSOV_S_A",20:"LAEVSKIY_V_M" }

def Mtxto2D(mtx,siz):
    arr=[]
    print "tr"
    for j in range(0,n):
                for i in range(0,n):
                    
                    arr.append(mtx[j][i])
    return arr

    def GenMtx():
        #makes kinda mtx of rands
            X=[]
            print "gen"
            mxl=[]
            for j in range(0,n):
                for i in range(0,n):
                    
                    mxl.append((random.randint(0,10)/10.0))
                X.append(mxl)
                mxl=[]
            return X


numf=30
y = []
n=100
Xlabel=[]
counter_ill=0
counter_healthy=0
for papka in range(1,kolpapok+1):
    for i in range(0,totalvid):
        Namefil=papki[papka]+'/'+NameStd+str(i)+".avi"
        if not os.path.isfile(Namefil):
            print "No file "+Namefil
            continue
        vc = cv2.VideoCapture((Namefil))
        print Namefil
        #numf shoud be func
        #taking shots with appropriate percent of black(bloody) pixels
        #kinda 15%, max% , 15%
        count=0.0
        pixcount =0.0
        percent =0.0
        r,f=vc.read()
        avg = np.float32(f)
        counter=0
        #numf should be func
       # vc.set(1,numf)    
        #print "set"
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
            #mask = cv2.inRange(f, 0, 130) # mask with 0-130 pixels intencity

           
        ##  inRange preobrazuet tsvetnuu v cherno beluu masku.
        ##  vse pixeli v zadannom diapazone teper belue. ostalnue - chernue


           # f = cv2.medianBlur(f,7) #  median blur 
          # medianny filtr(razmuvaet vso) 7 - vidimo radius razmyvania(kvadratom 7x7)

          #  cv2.accumulateWeighted(f,avg,0.05) # running average #Skolzashee srednee / vidimo zapis v avg/ 
           
          #^^^^ we keep feeding each frame to this function,
          #and the function keep finding the averages of all frames fed to it 
          
          #0.05 eto alfa .alpha is the weight of the input image. According to Docs, alpha regulates the update speed 
          #(how fast the accumulator “forgets” about earlier images). In simple words, 
          #if alpha is a higher value, average image tries to catch even very fast 
          #and short changes in the dfata. If it is lower value, average becomes sluggish 
          #and it won't consider fast changes in the input images.
          #fon vybiraet karoch
    ##        res = cv2.convertScaleAbs(avg) # scale and convert running average  
    ##        ff =f & mask # applying mas
    ##        resf = res & mask # applying mask k tomu che iz running avrg
    ##        cv2.subtract(resf,ff,new) 

            
    ##        ret,th4 = cv2.threshold(new,10,255,cv2.THRESH_TOZERO)
    ##        ret,th5 = cv2.threshold(th4,10,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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
            #print "NUMBER"
            #print str(vc.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
                
            #f=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
            #f=f/255.0
            #f=cv2.resize(f,(100,100),3)
           # print"normil"
            
            #cv2.imshow('l',th5)
            #cv2.imshow("Sobl",sobelx)
            #cv2.imshow("Sobly",sobely)
           # cv2.imshow("LAPLA",laplac)
            
           # cv2.imshow("LAPLAblur",dst/5)
          #  cv2.imshow("soblblur",dst2)
            
           # cv2.imshow("fin",tocount)
           # cv2.waitKey(1000)
            if vc.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)>=vc.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT ):
                break
            #class dat(DenseDesignMatrix):           
             #   def __init__(self):
            #self.class_names = ['0', '1']

        vc.set(1,int(numf))
        r,f=vc.read()
        #cv2.imshow("result",f)
        #cv2.waitKey(1000)
        #f=cv2.resize(f,(,224),3)
        return f
