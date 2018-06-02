# -*- coding: utf-8 -*-
import numpy as np

class CurveWrppaer(object):
    def __init__(self, get_df_functor):
        self.__functor = get_df_functor
    
    def get_df(self, t: float) -> float:
        return self.__functor(t)
