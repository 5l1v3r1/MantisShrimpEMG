# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 22:52:59 2018

@author: Dell
"""

from pypaths import Pypath

pp = Pypath()

linux_in = pp.to_native('~/asdf/asdf/asdf/')
print(linux_in)

win_in = pp.to_native(r'C:\Users\Dell\Downloads')
print(win_in)