# -*- coding: utf-8 -*-
'''
    geophpy.__config__
    ---------------------------

    Global variables used by geophpy.

    :copyright: Copyright 2014-2025 L. Darras, P. Marty, Q. Vitale and contributors, see AUTHORS.
    :license: GNU GPL v3.

'''

from IPython import get_ipython
import os
import platform
import warnings

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Variables à modifier pour rentrer les paramètres de votre config

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


### === AUTO DÉFINIES === ###
spyder = get_ipython().__class__.__name__ == 'SpyderShell'
os_kernel = platform.system()
### ===================== ###

### === Chemins === ###
# json_path : Chemin contenant les bases d'appareils/de constantes
# fortran_path : Chemin contenant les fichiers Fortran
_dir_name = os.path.dirname
abs_path = os.path.abspath(_dir_name(_dir_name(__file__)))+"/geophpy/"
emi_json_path = abs_path+"emi/__JSON__/"
emi_fortran_path = abs_path+"emi/__FORTRAN__/"
### =============== ###

### === Précisions === ###
prec_coos = 4 # Précision (chiffres après la virgule) des données "coordonnées" [4]
prec_data = 2 # Précision (chiffres après la virgule) des données "mesures" [2]
### ================== ###

### === MatPlotLib === ###
fig_width = 16 # Largeur des figures mpl [16]
fig_height = 9 # Hauteur des figures mpl [9]
fig_render_time = 0.25 # Temps (en s) consacré à charger la figure (uniquement celles en cours d'exécution) [0.25]
### ================== ###

### === Couleurs === ###
no_blink = False # Retirer l'effet clignotant du texte (pas toujours compatible avec le shell) [False]

base_color = '\33[0m'

bold_color = '\33[0;1m'
und_color ='\33[0;4m'
bold_und_color = '\33[0;1;4m'
if no_blink:
    blink_color = ''
else:
    blink_color = '\33[5m'
error_color = '\33[0;1;31m' 
warning_color = '\33[0;1;33m' 
if spyder:
    code_color = '\33[0;1;34m'
    success_color = '\33[0;1;32m'
    success_low_color = '\33[0;1;36m'
else:
    code_color = '\33[0;1;36m'
    success_color = '\33[0;1;92m'
    success_low_color = '\33[0;92m'
title_color = '\33[0;1;4;33m' 
title_next_color = '\33[0;33m' 
type_color = '\33[35m'
### ================ ###

### === Warnings === ###
no_warnings = True # Ne plus afficher les warnings externes au code (ceux des librairies) [True]
if no_warnings:
    warnings.filterwarnings("ignore")
### ================ ###



