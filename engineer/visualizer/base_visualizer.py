"""
@File: base_visualizer.
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2022-09-06
@Desc: This is Virtual Class to define Visualizer class
#
"""

class Base_Visualizer(object):
    def __init__(self):
        super(Base_Visualizer).__init__()

    def add_image(self, img):
        raise NotImplemented
    def add_scalar(self, img):
        raise NotImplemented
