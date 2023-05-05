"""
@File: base_optimzier.py
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2022-07-13
@Desc: base optimizer used to fit human garment
"""

import numpy as np
import torch
class _Base_Optimizer(object):
    def __init__(self, optimizer_setting):
        ''' Base Optimizer class
        Parameters
            optimizer_setting: optimizer used in this fitting process, e.g. Adam, SGD

        '''
        super(_Base_Optimizer, self).__init__()
        self.__name = '_Base_Optimizer'
        self.optimizier_setting = optimizer_setting
        self.input_para = dict()


    def solver(self, **kwargs):
        raise NotImplemented

    def fitting(self, inputs):
        raise NotImplemented

    def __call__(self, **inputs):
        return self.fitting(inputs)


    def get_optimzier(self, **parameters):
        raise NotImplemented


    def __repr__(self):
        __repr = "{}(Parameters: ".format(self.__name)
        for key in self.input_para.keys():
            __repr+="{}:{}, ".format(key,self.input_para[key])
        __repr=__repr[:-2]
        return __repr+')'

    @property
    def name(self):
        __repr = "{}".format(self.__name)
        return __repr

    @name.setter
    def name(self,v):
        self.__name = v
