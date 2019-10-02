# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 21:20:59 2019

@author: Rui Kong
"""
import FA
if __name__ == "__main__":

     bound = np.tile([[-600], [600]], 25)
     fa = FA(60, 25, bound, 200, [1.0, 0.000001, 0.6])
     fa.solve()