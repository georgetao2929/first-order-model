import matplotlib

matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
import skimage
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数

        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头

        self.set_ui()  # 初始化程序界面
        self.slot_init()  # 初始化槽函数

    '''程序界面布局'''

    def set_ui(self):
        self.__layout_main = QtWidgets.QVBoxLayout()  # 总布局
        self.__layout_fun_button = QtWidgets.QVBoxLayout()  # 按键布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # 数据(视频)显示布局
        self.__layout_data_show2 = QtWidgets.QVBoxLayout()  # 数据(视频)显示布局 added

        self.__layout_all_button_H=QtWidgets.QHBoxLayout()
        self.__layout_all_show_H=QtWidgets.QHBoxLayout()


        self.button_open_camera = QtWidgets.QPushButton('打开相机')  # 建立用于打开摄像头的按键
        self.button_close = QtWidgets.QPushButton('退出')  # 建立用于退出程序的按键
        self.button_resetframe = QtWidgets.QPushButton('重设')  # 建立用于重设帧的按键
        self.button_change_cource_image = QtWidgets.QPushButton('更换图片')
        self.button_open_camera.setMinimumHeight(50)  # 设置按键大小
        self.button_close.setMinimumHeight(50)
        self.button_resetframe.setMinimumHeight(50)
        self.button_change_cource_image.setMinimumHeight(50)

        # self.button_close.move(10, 100)  # 移动按键
        '''信息显示'''
        self.label_show_camera = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(640, 480)  # 给显示视频的Label设置大小为641x481
        self.label_show_camera2 = QtWidgets.QLabel()  # 定义显示视频的Label added
        self.label_show_camera2.setFixedSize(640, 480)  # 给显示视频的Label设置大小为641x481 added
        '''把按键加入到按键布局中'''
        self.__layout_all_button_H.addWidget(self.button_open_camera)  # 把打开摄像头的按键放到按键布局中
        self.__layout_all_button_H.addWidget(self.button_resetframe)  # 把重设帧的按键放到按键布局中
        self.__layout_all_button_H.addWidget(self.button_change_cource_image)
        self.__layout_all_button_H.addWidget(self.button_close)  # 把退出程序的按键放到按键布局中

        self.__layout_all_show_H.addWidget(self.label_show_camera)
        self.__layout_all_show_H.addWidget(self.label_show_camera2)
        '''把某些控件加入到总布局中'''

        self.__layout_main.addLayout(self.__layout_all_show_H)  # 把按键布局加入到总布局中
        self.__layout_main.addLayout(self.__layout_all_button_H)  # 把按键布局加入到总布局中
        # self.__layout_main.addWidget(self.label_show_camera)  # 把用于显示视频的Label加入到总布局中
        # self.__layout_main.addWidget(self.label_show_camera2)  # 把用于显示视频的Label加入到总布局中 added
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

        parser = ArgumentParser()
        parser.add_argument("--config", default='config/vox-256.yaml', help="path to config")
        parser.add_argument("--checkpoint", default='checkpoint/vox-cpk.pth.tar',
                            help="path to checkpoint to restore")
        parser.add_argument("--source_image", default='image/1.jpg', help="path to source image")
        parser.add_argument("--relative", dest="relative", action="store_true",
                            help="use relative or absolute keypoint coordinates")
        parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                            help="adapt movement scale based on convex hull of keypoints")
        parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                            help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")
        parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,
                            help="Set frame to start from.")
        parser.add_argument("--enc_downscale", default=1, type=float,
                            help="Downscale factor for encoder input. Improves performance with cost of quality.")
        parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.",default=False)
        parser.set_defaults(relative=True)
        parser.set_defaults(adapt_scale=False)
        self.opt = parser.parse_args()

        self.generator, self.kp_detector = self.load_checkpoints(self.opt.config, self.opt.checkpoint, self.opt.cpu)
        source_image = imageio.imread("image/jobs.jpg")
        self.source_image = resize(source_image, (256, 256))[..., :3]

        self.flag_first=0                       #用于判断是否需要向driving_video里插入一个初始帧
        self.frame_copy=0                       #用于拷贝一个初始帧
        self.x1=170                             #有效范围的左上角横坐标
        self.x2=470                             #有效范围的右下角横坐标
        self.y1=90                              #有效范围的左上角纵坐标
        self.y2=390                             #有效范围的右下角纵坐标



    '''初始化所有槽函数'''

    def slot_init(self):
        self.button_change_cource_image.clicked.connect(self.button_change_cource_image_clicked)
        self.button_resetframe.clicked.connect(self.button_reset_frame_clicked)
        self.button_open_camera.clicked.connect(
            self.button_open_camera_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()
        self.timer_camera.timeout.connect(self.show_animation)  # 若定时器结束，则调用show_camera() added
        self.button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序

    '''槽函数之一'''

    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(50)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('关闭相机')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.label_show_camera2.clear()  # 清空视频显示区域 added
            self.button_open_camera.setText('打开相机')
            self.flag_first=0

    def button_change_cource_image_clicked(self):
        directory = QtWidgets.QFileDialog.getOpenFileName(self,
                                                          "getOpenFileName", "./",
                                                          "All Files (*);;Text Files (*.txt)")
        # 当窗口非继承QtWidgets.QDialog时，self可替换成 None
        self.source_image = imageio.imread(directory[0])
        self.source_image = resize(self.source_image, (256, 256))[..., :3]

    def button_reset_frame_clicked(self):
        self.flag_first = 0

    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读取
        cv2.rectangle(self.image, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2)
        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage
        # self.label_show_camera2.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage added

    def show_animation(self):
        flag, self.frame = self.cap.read()  # 从视频流中读取
        driving_video = []
        if self.flag_first==0:
            self.frame_copy = self.frame
            driving_video.append(self.frame)
            driving_video.append(self.frame)
            self.flag_first=1
            return
        else:
            driving_video=[]
            driving_video.append(self.frame_copy)
            driving_video.append(self.frame)
            driving_video = [frame[..., ::-1] for frame in driving_video]
            driving_video = [frame[self.y1:self.y2, self.x1:self.x2] for frame in driving_video]
            driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

            self.predictions = self.make_animation(self.source_image, driving_video, self.generator,
                                                   self.kp_detector, relative=self.opt.relative,
                                                   adapt_movement_scale=self.opt.adapt_scale, cpu=self.opt.cpu)

            self.predictions = [frame[..., ::-1] for frame in self.predictions]

        self.predictions = [frame[..., ::-1] for frame in self.predictions]
        self.predictions = img_as_ubyte(self.predictions)
        self.predictions = cv2.resize(self.predictions[-1], (640, 480))
        showAnimation = QtGui.QImage(self.predictions.data, self.predictions.shape[1], self.predictions.shape[0],self.predictions.shape[1]*3,
                                 QtGui.QImage.Format_RGB888)
        self.label_show_camera2.setPixmap(QtGui.QPixmap.fromImage(showAnimation))


    def make_animation(self,source_image, driving_video, generator, kp_detector, relative, adapt_movement_scale,
                       cpu=False):
        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not cpu:
                source = source.cuda()
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector(driving[:, :, 0])

            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]
                if not cpu:
                    driving_frame = driving_frame.cuda()
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                       use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return predictions

    def load_checkpoints(self,config_path, checkpoint_path, cpu=False):
        with open(config_path) as f:
            config = yaml.load(f)

        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        if not cpu:
            generator.cuda()

        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params'])
        if not cpu:
            kp_detector.cuda()

        if cpu:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)

        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])

        if not cpu:
            generator = DataParallelWithCallback(generator)
            kp_detector = DataParallelWithCallback(kp_detector)

        generator.eval()
        kp_detector.eval()

        return generator, kp_detector

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = Ui_MainWindow()  # 实例化Ui_MainWindow
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过
