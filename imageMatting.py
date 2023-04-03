import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QComboBox, QApplication, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog
from pspnet import predict
import cv2
import os
from imageio import imread
from scipy.misc import imsave

VOC_CLASSES = ['飞机', '自行车', '鸟', '船',        # 这是本项目支持抠图元素的类别和颜色
               '瓶子', '公交车', '汽车', '猫', '椅子', '牛',
               '餐桌', '狗', '马', '摩托车', '人',
               '盆栽', '羊', '沙发', '火车', '电视、显示器']
VOC_COLORMAP = [[128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]


class Matting(QWidget):   # 使用pyqt5实现的一个界面
    def __init__(self):
        super().__init__()
        self.setWindowTitle('万能抠图神器')
        self.resize(1000, 600)
        VBox = QVBoxLayout(self)
        HBox = QHBoxLayout(self)
        pixmap = QPixmap("demo.jpg")
        self.imgName = "demo.jpg"
        self.leftLbl = QLabel(self)
       # self.leftLbl.resize(400, 400)
        self.leftLbl.setPixmap(pixmap)
        HBox.addWidget(self.leftLbl)
        pixmap = QPixmap("demo.jpg")
        self.imgName = "demo.jpg"
        self.image = ""
        self.rightLbl = QLabel(self)
       # self.rightLbl.resize(400, 400)
        self.rightLbl.setPixmap(pixmap)
        HBox.addWidget(self.rightLbl)
        VBox.addLayout(HBox)
        HBox = QHBoxLayout(self)
        btn1 = QPushButton("上传图片")
        btn1.clicked.connect(self.uploadImage)
        HBox.addWidget(btn1)
        self.cb = QComboBox(self)
        self.cb.addItems(VOC_CLASSES)
        HBox.addWidget(self.cb)
        btn2 = QPushButton("确定")
        btn2.clicked.connect(self.operation)
        HBox.addWidget(btn2)
        VBox.addLayout(HBox)
        self.setLayout(VBox)
        self.now_index = 0
        self.cb.currentIndexChanged[int].connect(self.changeValue)

    def changeValue(self, i):    # 用户选择类别
        self.now_index = i

    def uploadImage(self):   # 用户上传图片
        self.imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        self.image = QPixmap(self.imgName).scaled(self.leftLbl.width(), self.leftLbl.height())
        self.leftLbl.setPixmap(self.image)

    def operation(self):  # 抠图的具体实现过程
        if not os.path.exists('example_results'):
            os.mkdir('example_results')
        filename = self.imgName.split('/')[-1]
        output_path = 'example_results/' + filename
        print(output_path)
        #predict(self.imgName, output_path, 'pspnet101_voc2012')
        img1 = imread(self.imgName, pilmode='RGB')
        path1 = 'example_results/' + filename.split('.')[0] + '_seg.jpg'
        print(path1)
        img2 = imread(path1, pilmode='RGB')
        print(VOC_COLORMAP[self.now_index])
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                p1 = img2[i][j].tolist()
                if p1 == VOC_COLORMAP[self.now_index]:
                    print(p1)
                if p1 != VOC_COLORMAP[self.now_index]:
                    img1[i][j][0] = 255
                    img1[i][j][1] = 255
                    img1[i][j][2] = 255
        print('complete')
        imsave('example_results/result.jpg', img1)
        image = QPixmap('example_results/result.jpg').scaled(self.rightLbl.width(), self.rightLbl.height())
        self.rightLbl.setPixmap(image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    matting = Matting()
    matting.show()
    sys.exit(app.exec_())
