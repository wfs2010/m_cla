# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : 汪逢生
# @FILE     : crawl_music.py
# @Time     : 2020-12-13 下午 1:40
# @Software : PyCharm

import requests
import json

url ='https://api.i-meto.com/meting/api?server=netease&type=playlist&id=2142580576&r=0.8248415771905837'

html = requests.get(url,'lxml')
dict1 = json.loads(html.content)
with open('music.txt','w',encoding='utf-8') as f:
    for i in dict1:
        f.write(i['title']+'\n')