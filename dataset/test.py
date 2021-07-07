# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:44:54 2021

@author: paixo
"""

import os

arquivo = open('n_r3d_train.txt', 'r+')
#novo_arquivo = open("n_r3d_train.txt", "w+")

#print(len(arquivo.readlines()))

for linha in arquivo.readlines():
    filename = linha.split('\t')
    filename = filename[0]
    #print(filename)
    
    if os.path.isfile(filename[10:]):
        print(filename)
        #novo_arquivo.write(linha)
        
#novo_arquivo.close()
#arquivo.close()