# -*- coding: utf-8 -*-
'''
    geophpy.emi.json
    ---------------------------

    Intern json file management and related Fortran scripts for EM data.

    :copyright: Copyright 2014-2025 L. Darras, P. Marty, Q. Vitale and contributors, see AUTHORS.
    :license: GNU GPL v3.

'''

import json
import os
import subprocess
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

import geophpy.__config__ as CONFIG
from geophpy.core.operation import coeffs_relation
from geophpy.core.json import JSON_Indent_Encoder

def add_device(app_name,config,nb_channels,freq_list,gps=True,gps_dec=[0.0,0.0],
               TR_l=None,TR_t=None,height=0.1,bucking_coil=0,coeff_c_ph=None,
               coeff_c_qu=None,config_angles=None,save=True,error_code=False):
    """ 
    Create device with requested components, then save it in ``JSONs/Appareils.json``.\n
    TODO : Implement ``config_angles``.
    
    Notes
    -----
    Any device that shares all attributes with an existing entry will be ignored and raise a warning.
    
    Parameters
    ----------
    app_name : str
        Device name. Can be anything.
    config : str, {``"HCP"``, ``"VCP"``, ``VVCP``, ``"PRP_CS"``, ``"PRP_DEM"``, \
    ``"PAR"``, ``"COAX_H"``, ``"COAX_P"``, ``"CUS"``}
        Coil configuration.
    nb_channels : int
        Number of X and Y columns. The number of coils.
    freq_list : list of int
        Frequences of each coil. If all are the same, can be of length 1.
    ``[opt]`` gps : bool, default : ``True``
        If got GPS data.
    ``[opt]`` gps_dec : [float, float], default : ``[0.0,0.0]``
        Shift between the GPS antenna and the device center, on both axis (m).
        Should be ``[0,0]`` if none.
    ``[opt]`` TR_l : ``None`` or list of float, default : ``None``
        Distance between each coil and the transmitter coil, on lateral axis (m).
    ``[opt]`` TR_t : ``None`` or list of float, default : ``None``
        Distance between each coil and the transmitter coil, on transversal axis (m).
    ``[opt]`` height : float, default : ``0.1``
        Height of the device during the prospection (m).
    ``[opt]`` bucking_coil, default : ``0``
        Index of the bucking coil between coils (from ``1`` to ``nb_channels``).
        If none, set to 0.
    ``[opt]`` coeff_c_ph : ``None`` or list of float, default : ``None``
        Device constant given by the maker (in phase). 
        If ``None``, is set to an array of 1s.
    ``[opt]`` coeff_c_qu : ``None`` or list of float, default : ``None``
        Device constant given by the maker (in quadrature). 
        If ``None``, is set to an array of 1s.
    ``[opt]`` config_angles : ``None`` or list of [float, float], default : ``None``
        If ``config = "CUS"`` (custom), define the angles of each coil.
    ``[opt]`` save : bool, default : ``True``
        Saves new devices in json.
    ``[opt]`` error_code : bool, default : ``False``
        Instead of returning the dictionary, return an error code as an int.
    
    Returns
    -------
    * ``error_code = False``
        new_app : dict
            Output dictionary.
    * ``error_code = True``
        ec : int, {``0``, ``1``}
            Exit code (1 if ``new_app`` already exists)
    
    Raises
    ------
    * Unknown ``config``.
    * Lengths of ``TR_l``, ``TR_t``, ``coeff_c_ph`` and ``coeff_c_qu`` do not match ``nb_channels``.
    * None of ``TR_l`` and ``TR_t`` is specified.
    * ``config_angles = None`` even though``config = "CUS"``
    * Length of ``config_angles`` does not match ``nb_channels``.
    * Lenght of ``gps_dec`` is not equal to 2.
    """
    app_list = {}
    with open(CONFIG.emi_json_path+"Appareils.json", 'r') as f:
        # Chargement de la base actuelle
        app_list = json.load(f)
    
    ### À DÉCOMMENTER POUR RÉINITIALISER LE FICHIER ###
    # app_list ={
    #     "app_list": []
    #    }
    
    # Liste des configurations acceptées
    config_list = ["HCP","VCP","VVCP","PRP_CS","PRP_DEM","PAR","COAX_H","COAX_P","CUS"]
    # Pleins de checks pour s'assurer de la cohérence des données
    if config not in config_list:
        raise ValueError("Unknown configuration, must be in {}.".format(config_list))
    if TR_t == None and TR_l == None:
        raise ValueError("One of 'TR_L' or 'TR_t' must be defined.")
    if TR_l == None:
        TR_l = [0.0 for i in range(nb_channels)]
    if TR_t == None:
        TR_t = [0.0 for i in range(nb_channels)]
    if len(TR_l) != nb_channels:
        raise ValueError("Length of 'TR_l' does not match 'nb_channels' \
                         ({} and {}).".format(len(TR_l),nb_channels))
    if len(TR_t) != nb_channels:
        raise ValueError("Length of 'TR_t' does not match 'nb_channels' \
                         ({} and {}).".format(len(TR_t),nb_channels))
    if coeff_c_ph == None:
        coeff_c_ph = [1.0 for i in range(nb_channels)]
    if coeff_c_qu == None:
        coeff_c_qu = [1.0 for i in range(nb_channels)]
    if len(coeff_c_ph) != nb_channels:
        raise ValueError("Length of 'coeff_c_ph' does not match 'nb_channels' \
                         ({} and {}).".format(len(coeff_c_ph),nb_channels))
    if len(coeff_c_qu) != nb_channels:
        raise ValueError("Length of 'coeff_c_qu' does not match 'nb_channels' \
                         ({} and {}).".format(len(coeff_c_qu),nb_channels))
    
    # Création de l'appareil
    new_app = {}
    new_app["app_id"] = len(app_list["app_list"])
    new_app["app_name"] = app_name
    new_app["config"] = config
    if config == "CUS":
        if config_angles == None:
            raise ValueError("'config_angles' must be defined for 'CUS' configuration.")
        if len(config_angles) != 2*nb_channels:
            raise ValueError("Length of 'config_angles' does not match 2*'nb_channels' \
                             ({} and {}).".format(len(config_angles),2*nb_channels))
        new_app["config_angles"] = config_angles
    new_app["GPS"] = gps
    if gps_dec != None:
        if len(gps_dec) != 2:
            raise ValueError("Length of 'gps_dec' must be 2 ({}).".format(len(gps_dec)))
        new_app["GPS_dec"] = gps_dec
    new_app["nb_channels"] = nb_channels
    new_app["TR_l"] = TR_l
    new_app["TR_t"] = TR_t
    new_app["TR"] = [np.sqrt(TR_l[i]**2 + TR_t[i]**2) for i in range(nb_channels)]
    new_app["height"] = height
    new_app["freq_list"] = freq_list
    new_app["bucking_coil"] = bucking_coil
    new_app["coeff_c_ph"] = coeff_c_ph
    new_app["coeff_c_qu"] = coeff_c_qu
    
    # Si l'appareil existe déjà, on ne l'ajoute pas
    for app in app_list["app_list"]:
        if {i:new_app[i] for i in new_app if i!='app_id'} == {i:app[i] for i in app if i!='app_id'}:
            warnings.warn('Device ({}, {}) already added. No effect.'.format(app_name,config))
            if error_code:
                # Pas d'appareil ajouté
                return 1
            else:
                # Sortie de l'appareil
                return new_app
    # Sauvegarde automatique du nouvel appareil
    if save:
        app_list["app_list"].append(new_app)
    
    # Liste des appareils enregistré en .json
    with open(CONFIG.emi_json_path+"Appareils.json", "w") as f:
        json.dump(app_list, f, indent=2)
    if error_code:
        # Appareil ajouté
        return 0
    else:
        # Sortie de l'appareil
        return new_app
        

def find_device(uid):
    """ 
    Find the requested device in ``JSONs/Appareils.json`` from ``"app_id"``.\n
    
    Parameters
    ----------
    uid : int
        Device's ``"app_id"`` value.
    
    Returns
    -------
    app : dict
        Found device.
    
    Raises
    ------
    * Unknown ``uid``.
    """
    app_list = {} # Je pourrais l'enlever mais c'est un témoin de mon innocence
    with open(CONFIG.emi_json_path+"Appareils.json", 'r') as f:
        # Chargement des appareils
        app_list = json.load(f)
    
    # Nombre de caractères max du terminal
    try:
        nc = os.get_terminal_size().columns
    except OSError:
        nc = 50
    # On cherche l'appareil
    for app in app_list["app_list"]:
        # On l'a trouvé (ouf !)
        if app["app_id"] == uid:
            # On l'affiche
            print_device_selected(app,nc)
            # Cette bande sert à faire joli et c'est symétrique youpi !
            print(CONFIG.success_color+"-"*nc)
            print(CONFIG.base_color)
            return app
    
    # Pas trouvé (n'existe pas)
    raise ValueError("Device of 'uid' {} does not exist.".format(uid))


def remove_device(uid):
    """ 
    Remove the requested device in ``JSONs/Appareils.json`` from ``"app_id"``.\n
    
    Parameters
    ----------
    uid : int
        Device's ``"app_id"`` value.
    
    Raises
    ------
    * Unknown ``uid``.
    """
    app_list = {} # Je pourrais l'enlever mais c'est un témoin de mon innocence V2
    with open(CONFIG.emi_json_path+"Appareils.json", 'r') as f:
        # Chargement des appareils
        app_list = json.load(f)
    try:
        # Suppression (la première fois de ma vie que j'utilise del)
        del app_list["app_list"][uid]
    except IndexError:
        raise ValueError("Device of 'uid' {} does not exist.".format(uid))
    
    # On réattribue un identifiant à chaque appareil pour éviter les trous
    for ic, app in enumerate(app_list["app_list"]):
        app["app_id"] = ic
    
    # Enregistrement des appareils restants
    with open(CONFIG.emi_json_path+"Appareils.json", "w") as f:
        json.dump(app_list, f, indent=2)
    

def modify_device(uid,new_values_dict):
    """ 
    Modify requested parameters on the requested device in ``JSONs/Appareils.json`` from ``"app_id"``.\n
    Any parameter that is not specified will remain untouched, unless they contradict each other.
    
    Parameters
    ----------
    uid : int or dict
        Device's ``"app_id"`` value, or the loaded dictionary of device's parameters.
    new_values_dict : dict
        Dictionary of changes. Constructed as such :\n
        ``{"param_1" : value_1, "param_2" : value_2, [...]}``,\n
        where ``"param_1"`` [...] are keys from the device (``"config"``, ``"TR_l"`` [...]).
    
    Returns
    -------
    app_data : dict
        Modified device.
    
    Raises
    ------
    * Unknown ``uid``.
    * Lengths of ``TR_l``, ``TR_t``, ``coeff_c_ph`` and ``coeff_c_qu`` do not match ``nb_channels``.
    * No ``config_angles`` even though``config = "CUS"``
    * Length of ``config_angles`` does not match ``nb_channels``.
    * Lenght of ``gps_dec`` is not equal to 2.
    """
    if isinstance(uid, int):
        # Chargement de l'appareil
        app_data = find_device(uid)
    else:
        app_data = uid
    # Attribution des nouvelles valeurs des paramètres (et oui déjà)
    for key, value in new_values_dict.items():
        app_data[key] = value
        
    # Liste des configurations acceptées
    config_list = ["HCP","VCP","VVCP","PRP_CS","PRP_DEM","PAR","COAX_H","COAX_P","CUS"]
    # Pleins de checks pour s'assurer de la cohérence des données
    if app_data["config"] not in config_list:
        raise ValueError("Unknown configuration, must be in {}.".format(config_list))
    if app_data["TR_t"] == None and app_data["TR_l"] == None:
        raise ValueError("One of 'TR_L' or 'TR_t' must be defined.")
    if len(app_data["TR_l"]) != app_data["nb_channels"]:
        raise ValueError("Length of 'TR_l' does not match 'nb_channels' \
                         ({} and {}).".format(len(app_data["TR_l"]),app_data["nb_channels"]))
    if len(app_data["TR_t"]) != app_data["nb_channels"]:
        raise ValueError("Length of 'TR_t' does not match 'nb_channels' \
                         ({} and {}).".format(len(app_data["TR_t"]),app_data["nb_channels"]))
    if len(app_data["coeff_c_ph"]) != app_data["nb_channels"]:
        raise ValueError("Length of 'coeff_c_ph' does not match 'nb_channels' \
                         ({} and {}).".format(len(app_data["coeff_c_ph"]),app_data["nb_channels"]))
    if len(app_data["coeff_c_qu"]) != app_data["nb_channels"]:
        raise ValueError("Length of 'coeff_c_qu' does not match 'nb_channels' \
                         ({} and {}).".format(len(app_data["coeff_c_qu"]),app_data["nb_channels"]))
    if app_data["config"] == "CUS":
        try:
            if len(app_data["config_angles"]) != 2*app_data["nb_channels"]:
                raise ValueError("Length of 'config_angles' does not match 2*'nb_channels' \
                                 ({} and {}).".format(len(app_data["config_angles"]),2*app_data["nb_channels"]))
        except:
            raise ValueError("'config_angles' must be defined for 'CUS' configuration.")
    if len(app_data["GPS_dec"]) != 2:
        raise ValueError("Length of 'gps_dec' must be 2 ({}).".format(len(app_data["GPS_dec"])))
    app_data["TR"] = [np.sqrt(app_data["TR_l"][i]**2 + \
                              app_data["TR_t"][i]**2) \
                      for i in range(app_data["nb_channels"])]
    
    # Chargement de la base des appareils
    with open(CONFIG.emi_json_path+"Appareils.json", 'r') as f:
        app_list = json.load(f)
    
    # Enregistrement de l'appareil à la place de l'ancien
    for ic,app in enumerate(app_list["app_list"]):
        if app["app_id"] == uid:
            app_list["app_list"][ic] = app_data
            break
    
    # Enregistrement de la base
    with open(CONFIG.emi_json_path+"Appareils.json", "w") as f:
        json.dump(app_list, f, indent=2)
    
    return app_data


def print_devices(uid=None):
    """ 
    Print the requested device in ``JSONs/Appareils.json`` from ``"app_id"``.\n
    
    Parameters
    ----------
    ``[opt]`` uid : ``None`` or int, default : ``None``
        Device's ``"app_id"`` value. If ``None``, print all.
    
    Raises
    ------
    * Unknown ``uid``.
    """
    app_list = {} # Je pourrais l'enlever mais c'est un témoin de mon innocence V3
    with open(CONFIG.emi_json_path+"Appareils.json", 'r') as f:
        # Chargement des appareils
        app_list = json.load(f)
    
    print("")
    # Nombre de caractères max du terminal
    try:
        nc = os.get_terminal_size().columns
    except OSError:
        nc = 50
    # On affiche tout
    if uid == None:
        for app in app_list["app_list"]:
            print_device_selected(app,nc)
    # On affiche seulement celui qu'on veut
    else:
        try:
            print_device_selected(next(app for app in app_list["app_list"] if app["app_id"] == uid),nc)
        except:
            raise ValueError("Device of 'uid' {} does not exist.".format(uid))
    
    # Cette bande sert à faire joli et c'est symétrique youpi ! V2
    print(CONFIG.success_color+"-"*nc)
    print(CONFIG.base_color)


def print_device_selected(app,nc):
    """ 
    Notes
    -----
    Subfunction of ``print_devices``\n
    """
    # Bon bah là c'est assez explicite, j'affiche avec un swag inconmensurable
    print(CONFIG.success_color+"-"*nc)
    print(CONFIG.type_color+"{} : ".format(app["app_id"])+CONFIG.title_color+\
          "{} ({})".format(app["app_name"],app["config"]))
    print(CONFIG.success_low_color+"\tGPS : "+CONFIG.base_color+"{}".format(app["GPS"]))
    print(CONFIG.success_low_color+"\tNb T/R : "+CONFIG.base_color+"{}, ".format(app["nb_channels"]))
    print(CONFIG.success_low_color+"\tPos l : "+CONFIG.base_color+"{}, ".format(app["TR_l"])+\
          CONFIG.success_low_color+"Pos t : "+CONFIG.base_color+"{}".format(app["TR_t"]))
    print(CONFIG.success_low_color+"\tz : "+CONFIG.base_color+"{}, ".format(app["height"])+\
          CONFIG.success_low_color+"Frequences : "+CONFIG.base_color+"{}".format(app["freq_list"]))


def add_coeff_to_json(app_data):
    """ 
    Create dictionary with requested components about the device.\n
    If the parameters are already in the ``JSONs/Constantes.json`` file, then it simply takes the results.
    Otherwise, compute the modeling constants, then save it in the .json.\n
    
    Notes
    -----
    Computation is done by Fortran scripts.
    
    Parameters
    ----------
    app_data : dict
        Dictionary of device data.
    
    Returns
    -------
    dict_const : dict
        Dictionary with the initial parameters and the affiliated constants.
    
    Raises
    ------
    A lot, but everything is handled by ``try`` keywords.
    
    See also
    --------
    ``add_device, compute_new_const``
    """
    const_list = {}
    with open(CONFIG.emi_json_path+"Constantes.json", 'r') as f:
        # Chargement des constantes
        const_list = json.load(f)
    
    ### À DÉCOMMENTER POUR RÉINITIALISER LE FICHIER ###
    # const_list ={}
    
    # Création d'un champ vide pour la nouvelle constante
    new_const = {}
    new_const["config"] = [[app_data["config"], {}]]
    new_const["config"][0][1]["TR"] = [[app_data["TR"], {}]]
    new_const["config"][0][1]["TR"][0][1]["height"] = [[app_data["height"], {}]]
    new_const["config"][0][1]["TR"][0][1]["height"][0][1]["freq_list"] = [[app_data["freq_list"], {}]]
    
    # OK j'avoue j'assume moyen cette merde mais ça marche complètement
    # On parcourt un arbre où chaque embranchement est une valeur du paramètre de l'étage.
    # La structure a comme étage de haut en bas : "config", "TR", "height", "freq_list".
    # On commence à parcourir "config". Si la config est nouvelle, on calcule les coeffs par Fortran, 
    # puis on insère une branche contenant les trois autres paramètres et en feuille les valeurs des constantes.
    # Sinon, on se rend dans la config connue, et on cherche sur "TR" etc.
    # Si à la fin on a trouvé une occurence pour chacun des paramètres, ça veut dire que les constantes 
    # ont déjà été calculées pour cette appareil. On la prend et on insère rien dans le .json.
    # On trie par ordre alpha ou croissant la base pour la lisibilité (ne change pas la vitesse de recherche).
    try:
        ic1 = [e[0] for e in const_list["config"]].index(new_const["config"][0][0])
        #print("(1)", [e[0] for e in const_list["config"]])
        new_const_part = new_const["config"][0][1]
        const_list_part = const_list["config"][ic1][1]
        try:
            ic2 = [e[0] for e in const_list_part["TR"]].index(new_const_part["TR"][0][0])
            #print("(2)", [e[0] for e in const_list_part["TR"]])
            new_const_part = new_const_part["TR"][0][1]
            const_list_part = const_list_part["TR"][ic2][1]
            try:
                ic3 = [e[0] for e in const_list_part["height"]].index(new_const_part["height"][0][0])
                #print("(3)", [e[0] for e in const_list_part["height"]])
                new_const_part = new_const_part["height"][0][1]
                const_list_part = const_list_part["height"][ic3][1]
                try:
                    ic4 = [e[0] for e in const_list_part["freq_list"]].index(new_const_part["freq_list"][0][0])
                    #print("(4)", [e[0] for e in const_list_part["freq_list"]])
                    new_const_part = new_const_part["freq_list"][0][1]
                    const_list_part = const_list_part["freq_list"][ic4][1]
                    # Constante déjà calculée
                    new_const["config"][0][1]["TR"][0][1]["height"][0][1]["freq_list"][0][1] = const_list_part
                except ValueError:
                    new_const["config"][0][1]["TR"][0][1]["height"][0][1]["freq_list"][0][1] = compute_new_const(app_data)
                    const_list["config"][ic1][1]["TR"][ic2][1]["height"][ic3][1]["freq_list"]\
                        .append(new_const_part["freq_list"][0])
                    const_list["config"][ic1][1]["TR"][ic2][1]["height"][ic3][1]["freq_list"] = \
                        sorted(const_list["config"][ic1][1]["TR"][ic2][1]["height"][ic3][1]["freq_list"], 
                               key=lambda x: "".join(str(s) for s in x[0]))
            except ValueError:
                new_const["config"][0][1]["TR"][0][1]["height"][0][1]["freq_list"][0][1] = compute_new_const(app_data)
                const_list["config"][ic1][1]["TR"][ic2][1]["height"].append(new_const_part["height"][0])
                const_list["config"][ic1][1]["TR"][ic2][1]["height"] = \
                    sorted(const_list["config"][ic1][1]["TR"][ic2][1]["height"], 
                           key=lambda x: float(x[0]))
        except ValueError:
            new_const["config"][0][1]["TR"][0][1]["height"][0][1]["freq_list"][0][1] = compute_new_const(app_data)
            const_list["config"][ic1][1]["TR"].append(new_const_part["TR"][0])
            const_list["config"][ic1][1]["TR"] = \
                sorted(const_list["config"][ic1][1]["TR"], 
                       key=lambda x: "".join(str(s) for s in x[0])) # Petite technique pour trier des listes ;)
    except ValueError:
        new_const["config"][0][1]["TR"][0][1]["height"][0][1]["freq_list"][0][1] = compute_new_const(app_data)
        const_list["config"].append(new_const["config"][0])
        const_list["config"] = sorted(const_list["config"], key=lambda x: x[0])
    except KeyError or IndexError:
        warnings.warn('All computed constants are reset.')
        new_const["config"][0][1]["TR"][0][1]["height"][0][1]["freq_list"][0][1] = compute_new_const(app_data)
        const_list = new_const
           
    # Remise à un format plus lisible
    dict_const = {}
    dict_const["config"] = new_const["config"][0][0]
    dict_const["TR"] = new_const["config"][0][1]["TR"][0][0]
    dict_const["height"] = new_const["config"][0][1]["TR"][0][1]["height"][0][0]
    dict_const["freq_list"] = new_const["config"][0][1]["TR"][0][1]["height"][0][1]["freq_list"][0][0]
    dict_const["sigma_a_ph"] = new_const["config"][0][1]["TR"][0][1]["height"][0][1]["freq_list"][0][1]["sigma_a_ph"]
    #dict_const["Kph_a_ph"] = new_const["config"][0][1]["TR"][0][1]["height"][0][1]["freq_list"][0][1]["Kph_a_ph"]
    dict_const["sigma_a_qu"] = new_const["config"][0][1]["TR"][0][1]["height"][0][1]["freq_list"][0][1]["sigma_a_qu"]
    #dict_const["Kph_a_qu"] = new_const["config"][0][1]["TR"][0][1]["height"][0][1]["freq_list"][0][1]["Kph_a_qu"]    
    
    # Enregistrement des constantes
    with open(CONFIG.emi_json_path+"Constantes.json", "w") as f:
        json.dump(const_list, f, indent=None, cls=JSON_Indent_Encoder)
    
    #print(dict_const)
    return dict_const


def ball_calibr(ball_file,config,TR,radius,z,x_min,x_max,sep='\t',y=0,step=5,bucking_coil=0,plot=False):
    """ 
    Uses Fortran to estimate the device coefficient for quadrature.
    
    Parameters
    ----------
    ball_file : dict
        File containing the user measurements.
    config : str, {``"PRP_CS"``, ``"PRP_DEM"``, ``HCP``, ``"VCP"``}
        Coil configuration.
    TR : list of int
        Distance between each coil and the transmitter coil (in cm).
    radius : int
        Radius of the aluminium ball.
    z : int
        Offset for z axis (height).
    x_min : int
        Starting position between the device and the ball.
        Can be negative, lower than ``x_max``.
    x_max  : int
        Ending position between the device and the ball.
        Can be negative, higher than ``x_min``.
    ``[opt]`` y : int, default : ``0``
        Offset for y axis (lateral).
    ``[opt]`` step : int, default : ``5``
        Distance between two consecutive measures.
    ``[opt]`` bucking_coil : int, default : ``0``
        Index of the bucking coil between coils (from ``1`` to ``len(TR)``).
        If none, set to 0.
    ``[opt]`` plot : bool, default : ``False``
        Enables plotting comparing theorical and practical values.
    
    Returns
    -------
    coeff : list of float
        List of every computed coefficients in coil order.
    
    Notes
    -----
    The indents are mandatory for the Fortran executable.\n
    Each new line (except the last one) must end by ``\\x0d\\n``, 
    which is the Windows end of line + new line combination.\n
    All parameters must be int as float shoud not be read correctly by Fortran.\n
    If the ``config`` would be ``"VVCP"``, must instead specify ``"HCP"`` or ``"VCP"`` 
    depending on which methode has been used in the practical file.
    
    Raises
    ------
    * Unknown configuration.
    * Fortran executable fails.
    * Unknown OS.
    * ``x_min``, ``x_max`` and ``step`` do not match ``ball_file`` content.
    
    See also
    --------
    ``_file_constboule``
    """
    nb_channels = len(TR)
    # Listes des configurations proposées
    config_list = ["PRP_CS","PRP_DEM","HCP","VCP"]
    # Déso les gens mais ayez des vraies configurations aussi !
    if config not in config_list:
        raise NotImplementedError("Method unavailable for configuration '{}' \
                                  ({}).".format(config,config_list))
    
    # Répertoire de travail de l'utilisateur
    user_cwd = os.getcwd()
    
    # Nom des fichiers utilisés
    cfg_file = "_boule_.cfg"
    output_file = "_boule_.dat"
    fortran_exe = "boule.exe"
    fortran_linux = "boule.out"
    os.chdir(CONFIG.emi_fortran_path)
    
    # Création du fichier appelé par le Fortran
    _file_constboule(cfg_file,output_file,TR,radius,z,x_min,x_max,y,step,bucking_coil)
    
    # Appel du Fortran (en fonction des versions)
    if CONFIG.os_kernel == "Linux":
        error_code = subprocess.Popen(["./{}".format(fortran_linux),"{}".format(cfg_file)], 
                                      stdin=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                      stdout=subprocess.PIPE).communicate()[1]
    elif CONFIG.os_kernel == "Windows":
        error_code = subprocess.Popen(["start","{}".format(fortran_exe),"{}".format(cfg_file)], 
                                      stdin=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                      stdout=subprocess.PIPE).communicate()[1]
    elif CONFIG.os_kernel == "Darwin":
        error_code = subprocess.Popen(["./{}".format(fortran_exe),"-f","{}".format(cfg_file)], 
                                      stdin=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                      stdout=subprocess.PIPE).communicate()[1]
    else:
        raise NotImplementedError("Method unavailable for OS '{}'.".format(CONFIG.os_kernel))
    if error_code:
        raise Exception("Fortran error.")
    
    # Lecture des données du tableau
    don = pd.read_csv(output_file,sep='\s+',skiprows=1)
    config_index = next(i for i,c in enumerate(config_list) if c == config)
    cols_th = don.iloc[:,1+nb_channels*config_index:1+nb_channels*(config_index+1)]
    # Lecture des données du fichier de boule
    os.chdir(user_cwd)
    don = pd.read_csv(ball_file,sep=sep)
    don.drop(don[don[don.columns[0]] < 0].index, inplace=True)
    cols_pr = don[[c for c in don.columns if "Inph" in c]]
    
    # Affichage et calcul des constantes
    coeff = []
    try:
        for e in range(nb_channels):
            # Unité en ppm
            c_pr = cols_pr.iloc[:,e]*1000
            c_th = cols_th.iloc[:,e]
            if plot:
                fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(CONFIG.fig_width,CONFIG.fig_height))
                ax.plot(c_pr,c_th,'o')
                ax.set_title("Positions des bobines (y={}, z={})".format(y,z))
                ax.set_xlabel("Pratique")
                ax.set_ylabel("Théorique")
                plt.show()
            # Loi linéaire entre la théorie et la pratique
            lin_reg = linregress(c_pr,c_th)
            coeff.append(lin_reg.slope)
    except ValueError:
        raise ValueError("One of 'step' ({}), 'x_min' ({}) or 'x_max ({}) is incorrect"\
                         .format(step,x_min,x_max))
    except TypeError:
        raise LookupError("File is not read correctly.")
    
    # On retourne les coeffs
    return coeff


def _file_constboule(cfg_file,output_file,TR,radius,z,x_min,x_max,y=0,step=5,bucking_coil=0):
    """ 
    Construct a file that contains the mandatory data for the Fortran ball calibration procedure.\n
    
    Parameters
    ----------
    cfg_file : str
        File that will contain the informations for the Fortran file.
    output_file : str
        File that will contain the dataframe of theoretical values.
    TR : list of int
        Distance between each coil and the transmitter coil (in cm).
    radius : int
        Radius of the aluminium ball.
    z : int
        Offset for z axis (height).
    x_min : int
        Starting position between the device and the ball. Can be negative, lower than ``x_max``.
    x_max  : int
        Ending position between the device and the ball. Can be negative, higher than ``x_min``.
    ``[opt]`` y : int, default : ``0``
        Offset for y axis (lateral).
    ``[opt]`` step : int, default : ``5``
        Distance between two consecutive measures.
    ``[opt]`` bucking_coil : int, default : ``0``
        Index of the bucking coil between coils (from ``1`` to ``len(TR)``). If none, set to 0.
    
    Notes
    -----
    Subfunction of ``ball_calibr``.\n
    The indents are mandatory for the Fortran executable.\n
    Each new line (except the last one) must end by ``"\\x0d\\n"``, 
    which is the Windows end of line + new line combination.\n
    All parameters must be int as float shoud not be read correctly by Fortran.
    
    See also
    --------
    ``ball_calibr``
    """
    # Ce fichier est assez trivial on va pas se mentir
    with open(cfg_file, 'w') as f:
        f.write(str(len(TR))+"\x0d\n")
        f.write(str(bucking_coil)+"\x0d\n")
        f.write(output_file+"\x0d\n")
        for e in TR:
            f.write(str(e)+"\x0d\n")
        f.write(str(y)+" "+str(x_min)+" "+str(z)+"\x0d\n")
        f.write(str(step)+" "+str(x_max)+"\x0d\n")
        f.write(str(radius)+"")


def compute_new_const(app_data,plot=False):
    """ 
    Uses Fortran to compute a dataframe of constants values.\n
    Then estimates the coefficients of a linear of polynomial relation for each coil.
    
    Parameters
    ----------
    app_data : dict
        Dictionary of device data.
    ``[opt]`` plot : bool, default : ``False``
        Enables plotting of found linear/polynomial models.
    
    Returns
    -------
    const_dict : dict
        Dictionary of constants.
    
    Notes
    -----
    Subfunction of ``add_coeff_to_json``.\n
    macOS (``Darwin``) has not been tested yet (no Apple device).
    
    Raises
    ------
    * Fortran executable fails.
    * Unknown OS.
    
    See also
    --------
    ``add_coeff_to_json, _file_constab, coeffs_relation``
    """
    # Répertoire de travail de l'utilisateur
    user_cwd = os.getcwd()
    
    # Nom des fichiers utilisés
    cfg_file = "_config_.cfg"
    fortran_exe = "terrainhom.exe"
    fortran_linux = "terrainhom.out"
    os.chdir(CONFIG.emi_fortran_path)
    const_dict = {"sigma_a_ph": [], "sigma_a_qu": []}
    # Nombre de variation pour deux paramètres
    v = 100
    # Pour chaque voie
    for e in range(app_data["nb_channels"]):
        
        # Création du fichier appelé par le Fortran
        _file_constab(app_data,cfg_file,e,variation=v,
                      F_rau=1001,F_eps_r=None,F_kph=0.01,F_kqu=None)
        
        # Appel du Fortran (en fonction des versions)
        if CONFIG.os_kernel == "Linux":
            error_code = subprocess.Popen(["./{}".format(fortran_linux),"{}".format(cfg_file)], 
                                          stdin=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                          stdout=subprocess.PIPE).communicate()[1]
        elif CONFIG.os_kernel == "Windows":
            error_code = subprocess.Popen(["start","{}".format(fortran_exe),"{}".format(cfg_file)], 
                                          stdin=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                          stdout=subprocess.PIPE).communicate()[1]
        elif CONFIG.os_kernel == "Darwin":
            error_code = subprocess.Popen(["./{}".format(fortran_exe),"-f","{}".format(cfg_file)], 
                                          stdin=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                          stdout=subprocess.PIPE).communicate()[1]
        else:
            raise NotImplementedError("Method unavailable for OS '{}'.".format(CONFIG.os_kernel))
        if error_code:
            raise Exception("Fortran error.")
        
        # Lecture du tableau
        don = pd.read_csv(cfg_file[:-4]+".dat",sep='\s+',header=None)
        
        # unité en SI
        HsHp_qu = np.array(don.iloc[:,5])
        sigma = np.array(1/don.iloc[:,0])
        # Relation cubique
        saqu = coeffs_relation(HsHp_qu,sigma,m_type="poly_3",plot=plot)
        const_dict["sigma_a_qu"].append(saqu)
        
        # unité en SI
        HsHp_ph = np.array(don.iloc[:,4])
        HsHp_ph_corr = HsHp_ph.copy()
        # Retrait de l'effet de sigma en se reportant sur la première courbe qui passe par l'origine
        last_bar = (v-1)*v
        for i in range(last_bar+v):
            HsHp_ph_corr[i] -= HsHp_ph_corr[last_bar+i%v]
        # Relation cubique aussi
        saph = coeffs_relation(sigma,HsHp_ph_corr,m_type="poly_3",choice=False,plot=plot)
        const_dict["sigma_a_ph"].append(saph)
    
    os.chdir(user_cwd)
    return const_dict


def _file_constab(app_data,cfg_file,e,variation=None,
                  S_rau=1,S_eps_r=1,S_kph=0.1E-5,S_kqu=0.1E-7,
                  F_rau=None,F_eps_r=None,F_kph=None,F_kqu=None):
    """ 
    Construct a file that contains the mandatory data for the Fortran ppt procedure.\n
    
    Parameters
    ----------
    app_data : dict
        Dictionary of device data.
    cfg_file : str
        File that will contain the informations for the Fortran file.
    e : int
        Index of the chosen coil (the procedure manage one coil at a time).
    ``[opt]`` variation : ``None`` or int, default : ``None``
        Number of different values for each moving parameter. The maximum values are
            * 1 parameter : ``10000``.
            * 2 parameters : ``100``.
            * 3 parameters : ``21``.
            * 4 parameters : ``10``.
    ``[opt]`` S_rau : int, default : ``1``
        Starting value for :math:`\\rho`.
    ``[opt]`` S_eps_r : int, default : ``1``
        Starting value for :math:`\\epsilon_{r}`.
    ``[opt]`` S_kph : int, default : ``0``
        Starting value for :math:`\\kappa_{ph}`.
    ``[opt]`` S_kqu : int, default : ``0``
        Starting value for :math:`\\kappa_{qu}`.
    ``[opt]`` F_rau : ``None`` or int, default : ``None``
        Ending value for :math:`\\rho`.
        If ``None``, :math:`\\rho` is constant.
    ``[opt]`` F_eps_r : ``None`` or int, default : ``None``
        Ending value for :math:`\\epsilon_{r}`.
        If ``None``, :math:`\\epsilon_{r}` is constant.
    ``[opt]`` F_kph : ``None`` or int, default : ``None``
        Ending value for :math:`\\kappa_{ph}`. 
        If ``None``, :math:`\\kappa_{ph}` is constant.
    ``[opt]`` F_kqu : ``None`` or int, default : ``None``
        Ending value for :math:`\\kappa_{qu}`. 
        If ``None``, :math:`\\kappa_{qu}` is constant.
    
    Notes
    -----
    Subfunction of ``compute_new_const``.\n
    The indents are mandatory for the Fortran executable.\n
    Each new line (except the last one) must end by ``\\x0d\\n``, 
    which is the Windows end of line + new line combination.\n
    At least one of the 4 ``F_{arg}`` should be set.\n
    TODO : Custom configuration.
    
    Raises
    ------
    * Unknown configuration.
    * Values are too big (goes beyond the maximum allocated space).
    * No parameters are marked as varying.
    * The number of computation lines goes beyond 10000.
    
    See also
    --------
    ``compute_new_const``
    """
    # Angles des configs selon l'ordre donné en dessous
    # Si les valeurs puent la merde c'est que je ne les ai pas
    _Config_params = [[1.2,3.4,5.6,7.8],[0.0,0.0,0.0,90.0],[0.0,35.0,0.0,35.0],
                      [0.0,90.0,0.0,90.0],[0.0,0.0,0.0,0.0],[90.0,0.0,90.0,0.0],
                      [1.2,3.4,5.6,7.8],[1.2,3.4,5.6,7.8],[0.0,90.0,0.0,0.0],
                      [90.0,0.0,90.0,0.0]]
    
    # Là on a de quoi avoir peur...
    # Pour résumer, faut gérer l'indentation et c'est un enfer
    with open(cfg_file, 'w') as f:
        f.write(" # Geometrie(s) d'appareil\x0d\n")
        f.write(" {}\x0d\n".format(1))
        config_list = ["CUS","PRP_CS","PAR","HCP","COAX_H","VCP","COAX_V","VVCP","PRP_DEM"]
        try:
            config_id = next(i for i,c in enumerate(config_list) if c == app_data["config"])
        except StopIteration:
            raise ValueError("Unknown configuration, must be in {}.".format(config_list))
        nb_spaces = 4 - len(str(config_id)) # Est toujours égal à 1 pour l'instant
        f.write(" "*nb_spaces+"{}\x0d\n".format(config_id))
        f.write(" "*(4 - len(str(int(app_data["TR"][e]))))+\
                "{:.2f}\x0d\n".format(app_data["TR"][e]))
        f.write("   1.00\x0d\n") # Euh je sais plus en vrai
        f.write(" "*(4 - len(str(int(app_data["height"]))))+\
                "{:.2f}\x0d\n".format(app_data["height"]))
        f.write(" "*(4 - len(str(int(app_data["height"]))))+\
                "{:.2f}\x0d\n".format(app_data["height"]))
            
        f.write(" # Parametres utiles pour CSTM\x0d\n")
        angles = _Config_params[config_id]
        for i in range(4):
            nb_spaces = 4 - len(str(int(angles[i])))
            f.write(" "*nb_spaces+"{:.2f}\x0d\n".format(angles[i]))
        f.write(" "*(4 - len(str(int(app_data["TR_l"][e]))))+\
                "{:.2f}\x0d\n".format(app_data["TR_l"][e]))
        f.write(" "*(4 - len(str(int(app_data["TR_t"][e]))))+\
                "{:.2f}\x0d\n".format(app_data["TR_t"][e]))
        
        f.write(" # Tableau de frequence(s)\x0d\n")
        nb_spaces = 2 - len(str(len(app_data["freq_list"])))
        if nb_spaces < 0:
            raise ValueError("Lenght of 'freq_list' can't go past 99 \
                             ({}).".format(len(app_data["freq_list"])))
        f.write(" "*nb_spaces+"{}\x0d\n".format(len(app_data["freq_list"])))
        for freq in app_data["freq_list"]:
            nb_spaces = 9 - len(str(int(freq)))
            if nb_spaces < 0:
                raise ValueError("No frequence can't go past 9999999 Hz ({}).".format(freq))
            f.write(" "*nb_spaces+"{:.2f}\x0d\n".format(freq))
        
        f.write(" # Terrain 1D\x0d\n")
        nb_var = int(F_rau!=None)+int(F_eps_r!=None)+int(F_kph!=None)+int(F_kqu!=None)
        if nb_var == 0:
            raise ValueError("At least one of rau, eps, kph or kqu must vary.")
        f.write(" "+"{}\x0d\n".format(nb_var))
        limit_var = [10000,100,21,10]
        if variation == None:
            variation = limit_var[nb_var-1]
        else:
            if variation > limit_var[nb_var-1]:
                raise ValueError("Number of iterations can't go past \
                                 {}.".format(limit_var[nb_var-1]))
        nb_spaces = 5 - len(str(variation))
        f.write(" "*nb_spaces+"{}\x0d\n".format(variation))
        S_var_list = [S_rau,S_eps_r,S_kph,S_kqu]
        F_var_list = [F_rau,F_eps_r,F_kph,F_kqu]
        nb_spaces_var = [6,9,4,4]
        for ic,p in enumerate(F_var_list):
            if p != None:
                f.write(" "*3+"{}".format(ic+1))
        f.write("\x0d\n")
        f.write(" 2\x0d\n")
        for ic,p in enumerate(S_var_list):
            if ic <= 1:
                nb_spaces = nb_spaces_var[ic] - len(str(int(p)))
                if nb_spaces < 0:
                    raise ValueError("Starting value '{}' is too big.".format(p))
                f.write(" "*nb_spaces+"{:.2f}".format(p))
            else:
                f.write(" "*nb_spaces_var[ic]+"{:.5E}".format(p))
        f.write("\x0d\n")
        for ic,p in enumerate(F_var_list):
            if p != None:
                if ic <= 1:
                    nb_spaces = nb_spaces_var[ic] - len(str(int(p)))
                    if nb_spaces < 0:
                        raise ValueError("Ending value '{}' is too big.".format(p))
                    f.write(" "*nb_spaces+"{:.2f}".format(p))
                else:
                    f.write(" "*nb_spaces_var[ic]+fortran_sc_notation(p,5))
            else:
                if ic <= 1:
                    nb_spaces = nb_spaces_var[ic] - len(str(int(S_var_list[ic])))
                    f.write(" "*nb_spaces+"{:.2f}".format(S_var_list[ic]))
                else:
                    f.write(" "*nb_spaces_var[ic]+fortran_sc_notation(S_var_list[ic],5))
        f.write("\x0d\n")

def fortran_sc_notation(n,p):
    """ 
    Convert float to a Fortran readable scientific notation.\n
    Format : ``0.[value]E[+/-][exponent]``, example : ``0.12030E-03``.\n
    
    Parameters
    ----------
    n : float
        Float number to convert
    p : int
        Float precision.
    
    Returns
    -------
    s_res : str
        Format string 
    
    Notes
    -----
    Subfunction of ``_file_constab``.\n
    Python's scientific notation is shifted to the left and cannot be used as raw.
    """
    # Fortran utilise un truc qui n'existe pas en python tellement il est vieux
    # C'est comme la notation scientifique mais décalé de 1.
    s = "{:.{}E}".format(n,p)
    s_res = "0."+s[0]+s[2:-5]+"E"+"{:.{}E}".format(n*10,p)[-3:]
    return s_res