#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Proxy generator
Check module at s0lst1c3/grey_harvest.
"""

import grey_harvest
import numpy as np

harvester = grey_harvest.GreyHarvester() #spawn a harvester

#harvest some proxies from teh interwebz
count = 0
proxy_len = 200 #number of proxies to generate
proxy_list = []
for proxy in harvester.run():
        print proxy
        proxy_list.append(proxy)
        count += 1
        if count >= proxy_len:
                break
np.savetxt('proxylist2.txt', proxy_list, fmt='%s')