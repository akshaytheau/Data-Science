#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:00:43 2021

@author: akshay
"""

from gingerit.gingerit import GingerIt

text = 'Narendra Modi is our prme mnister. He is from Gujaratt'

parser = GingerIt()
print(len(parser.parse(text)['corrections']))
