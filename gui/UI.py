import sys, os
import cv2
#import frame_selector
#import neuro_tools
from PyQt4 import QtCore, QtGui, uic

class DD_MainWindow(QtGui.QWidget):
    def __init__(self, parent=None):

        QtGui.QWidget.__init__(self,parent)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Preferred)
        self.setMinimumSize(680,680)
        self.setMaximumSize(690,690)

        self.picture=QtGui.QPixmap()
        self.picture.load(QtCore.QString("heart.jpg"))
        #QLineEdit for Path
        self.pathEdit=QtGui.QLineEdit("video.avi")
        
        #Buttons
        self.analyze=QtGui.QPushButton(QtCore.QString("Analyze"),self)
        self.analyze.clicked.connect(self.Analyzer)
        
        self.download=QtGui.QPushButton(QtCore.QString("Download"),self)
        self.download.clicked.connect(self.Download)

        #Labels
        self.picLbl=QtGui.QLabel(self)
        self.picLbl.setAlignment(QtCore.Qt.AlignCenter)
        self.picLbl.setMinimumSize(565,584)
        self.picLbl.setMaximumSize(575,594)
        self.picLbl.setPixmap(self.picture)
        
        self.PathLabel=QtGui.QLabel("Path to video for analizys")

        #set layout
        topLayout=QtGui.QVBoxLayout(self)
        #topLayout.addWidget(self.XXX)
        layout1=QtGui.QHBoxLayout(self)
        
        layout1.addWidget(self.PathLabel)
        layout1.addWidget(self.pathEdit)
        
        layout2=QtGui.QHBoxLayout(self)
        layout2.addWidget(self.download)
        layout2.addWidget(self.analyze)
        
        topLayout.addWidget(self.picLbl)
        topLayout.addLayout(layout1)
        topLayout.addLayout(layout2)
        self.setLayout(topLayout)
        
    def Download(self):
        print "Download"
        self.picture.load(QtCore.QString("Kill_yourself.jpg"))
        self.picLbl.setPixmap(self.picture)
    def Analyzer(self):
        print "Analyzer"

def main():
    app=QtGui.QApplication(sys.argv)
    window=DD_MainWindow()

    palette	= QtGui.QPalette()
    palette.setBrush(QtGui.QPalette.Background,QtGui.QBrush(QtGui.QPixmap("bg.jpg"))) 
    window.setPalette(palette)
    window.setWindowTitle("DeepDiagnostics")
    window.show()
    # It's exec_ because exec is a reserved word in Python
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    

    
        
