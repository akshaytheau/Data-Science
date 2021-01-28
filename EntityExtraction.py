#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:39:18 2021

@author: akshay
"""

import en_core_web_sm
nlp = en_core_web_sm.load()
doc = nlp(u"Ramesh is earning in 100 dollars in UK")
for entity in doc.ents:
  print(entity.label_, ' | ', entity.text)
