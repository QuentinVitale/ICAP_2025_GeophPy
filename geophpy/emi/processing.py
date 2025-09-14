# -*- coding: utf-8 -*-
'''
    geophpy.emi.processing
    ----------------------------

    DataSet Object electric processing routines.

    :copyright: Copyright 2014-2019 Lionel Darras, Philippe Marty, Quentin Vitale and contributors, see AUTHORS.
    :license: GNU GPL v3.

'''
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import geophpy.core.utils as gutils
import geophpy.core.operation as goper
import geophpy.emi.operation as emioper
import geophpy.emi.json as emijson
import geophpy.__config__ as CONFIG

import os
import subprocess
import re
import warnings

def init_process(uid,file_list=None,sep='\t',sup_na=True,regr=False,l_r=None,corr_base=False,
                 no_base=False,pseudo_prof=False,l_p=None,plot=False,in_file=False,full_infos=False):
    """
    Apply to dataframe the first steps of data processing.\n
    1) Time correction, if GPS.\n
    2) Profile / base detection.\n
    3) Coordinates interpolation.\n
    4) ``[opt]`` ``NaN`` completion and profile linearization.\n
    5) Coil / GPS offsets.
    
    Parameters
    ----------
    uid : int or dict
        Device's ``"app_id"`` value, or the loaded dictionary of device's parameters.
    ``[opt]`` file_list : ``None`` or (list of) str or (list of) dataframe, default : ``None``
        List of files or loaded dataframes to process, or a single one. 
        If ``None``, takes all files from current working directory.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` sup_na : bool, default : ``True``
        If ``NaN`` completion is done.
    ``[opt]`` regr : bool, default : ``False``
        If profile linearization is done.
    ``[opt]`` l_r : ``None`` or list of int or str, default : ``None``
        List of decisions for profile linearization. 
        If ``None``, enables a choice procedure.
    ``[opt]`` corr_base : bool, default : ``False``
        If base correction is done.
    ``[opt]`` no_base : bool, default : ``False``
        If no file contains any bases, while having clear profiles.
    ``[opt]`` pseudo_prof : bool, default : ``False``
        If the prospection is represented by one continuous line.
    ``[opt]`` l_p : ``None`` or list of [float, float], default : ``None``
        List of points coordinates for segments. 
        If ``[]``, perform a linear regression instead. 
        If ``None``, enables a choice procedure.
    ``[opt]`` plot : bool, default : ``False``
        Enables plotting.
    ``[opt]`` in_file : bool, default : ``False``
        If ``True``, save result in a file. If ``False``, return the dataframe.
    ``[opt]`` full_infos : bool, default : ``False``
        If ``True``, return more variables (used for ``main``).
    
    Returns
    -------
    * ``in_file = True``
        none, but save dataframe for profiles and bases in separated .dat
    * ``in_file = False``
        * ``full_infos = False``
            ls_base : list of dataframe
                List of bases for each file
            ls_mes : list of dataframe
                List of profiles for each file
        * ``full_infos = True``
            ls_base : list of dataframe
                List of bases for each file
            ls_mes : list of dataframe
                List of profiles for each file
            ncx : list of str
                Names of every X columns.
            ncy : list of str
                Names of every Y columns.
            nc_data : list of str
                Names of every Z columns (actual data).
            nb_res : int
                The number of data per coil.
            ls_pd_done_before : list of dataframe
                List of all dataframe that were already processed by a previous function call.
    
    Notes
    -----
    This function ignores any dataframe that was already processed by a previous function call.\n
    Can plot data.\n
    If the prospection is done without GPS, ``no_base = True`` anyways.\n
    If the prospection is continuous in time (no jumps),
    ``no_base = True`` and ``pseudo_prof = True`` anyways.
    
    Raises
    ------
    * ``regr = True`` and ``l_r`` is in incorrect format.
    
    See Also
    --------
    ``check_time_date, time, detect_chgt, intrp_prof, detect_base_pos, detect_profile_square,
    XY_Nan_completion, sep_BP, detect_pseudoprof, synth_base, pts_rectif, evol_profiles, dec_channels``
    """
    # Conversion en liste si 'file_list' ne l'est pas
    try:
        if file_list != None and not isinstance(file_list,list):
            file_list = [file_list]
        file_list = gutils.true_file_list(file_list)
    # Type dataframe
    except ValueError:
        file_list = [file_list]
    # Chargement de l'appareil si 'uid' est l'indice dans la base 'Appareils.json'
    if isinstance(uid,int):
        app_data = emijson.find_device(uid)
    else:
        app_data = uid
    
    # Concaténation si nécessaire avant traitement
    is_loaded = False
    ls_pd=[]
    ls_pd_done_before = []
    nb_file = len(file_list)
    for ic,f in enumerate(file_list) :
        if isinstance(f,str):
            # Chargement des données
            data = gutils.check_time_date(f,sep)
        else:
            data = f
            is_loaded = True
        # Numérotation des fichiers
        data['File_id']=ic+1
        try:
            # Si la colonne 'X_int' existe déjà, le fichier est déjà interpolé (on l'ignore)
            data["X_int"]
            ls_pd_done_before.append(data)
        except KeyError:
            ls_pd.append(data)
    
    nb_f = len(ls_pd)
    nb_res = 2
    const_GPS = 2
    # GPS ou non
    if app_data["GPS"] :
        n_col_X='Easting'
        n_col_Y='Northing'
        for df in ls_pd:
            temp = df.dropna(subset=[n_col_X,n_col_Y])
            if temp[n_col_X].median() > temp[n_col_Y].median():
                ls_pd = gutils.switch_cols(ls_pd,n_col_X,n_col_Y)
    else :
        n_col_X='x[m]'
        n_col_Y='y[m]'
    if nb_f == 0:
        cdata = ls_pd_done_before[0]
    else:
        cdata = ls_pd[0]
    # On calcule le nombre de colonnes absentes avant les données (Cond/Inph) parmi celles potentielles.
    for c in ["Altitude","Date","Time","DOP","Satelites"]:
        try:
            cdata[c]
            const_GPS += 1
        except KeyError:
            pass
    
    # Initialisation des colonnes position et données
    ls_base = []
    ls_mes = []
    col_z=[const_GPS+i for i in range(app_data["nb_channels"]*nb_res)]
    ncx = ["X_int_"+str(e+1) for e in range(app_data["nb_channels"])]
    ncy = ["Y_int_"+str(e+1) for e in range(app_data["nb_channels"])]
    
    # Affichage des nom des fichiers traîtés, si on les a en entrée
    if not is_loaded:
        for ic,don_c in enumerate(ls_pd) :
            print("File n°{} : '{}'".format(ic+1,file_list[ic]))
     
    # Si au moins un fichier est à traîter, on effectue les étapes
    if nb_f != 0:
        don_raw = pd.concat(ls_pd)
        don_raw.index=np.arange(don_raw.shape[0])
        # Au cas où certains points ont des mesures manquantes,
        # on les enlève pour ne pas polluer le reste.
        don_raw.dropna(subset = don_raw.columns[col_z],inplace=True)
        don_raw.reset_index(drop=True,inplace=True)

        # Si le fichier contient des données temporelles
        try:
            # On gère les prospections faites des jours différents
            don_raw['Seconds'] = goper.set_time(don_raw)
            for ic,date_c in enumerate(don_raw['Date'].unique()) :
                if ic>0 :
                    ind_d =don_raw.index[don_raw['Date']==date_c]
                    don_raw.loc[ind_d,'Seconds']=don_raw.loc[ind_d,'Seconds']+ic*86400
        except KeyError:
            pass
        
        # Détection profils et interpolation des positions répétées
        if app_data["GPS"]:
            don_d = goper.detect_chgt(don_raw,[n_col_X,n_col_Y])
            don_i = goper.intrp_prof(don_d,n_col_X,n_col_Y)
            # Si le fichier ne contient pas de base, on ne considère que des profils
            if no_base:
                don_i["Base"] = 0
                don_i["Profil"] = don_i["B+P"]
            else:
                don_i = goper.detect_base_pos(don_i,2)
        else:
            don_raw["X_int"] = don_raw.iloc[:,0]
            don_raw["Y_int"] = don_raw.iloc[:,1]
            don_i = goper.detect_profile_square(don_raw)
        
        # Suppression ou completion des données manquantes
        if sup_na:
            don_i.dropna(subset = [n_col_X,n_col_Y,"X_int","Y_int"],inplace=True)
            don_i.reset_index(drop=True,inplace=True)
        else:
            # Si aucun profil n'a pu être détecté (pas de saut temporel, prospection continue),
            # on utilise un autre algo de complétion
            if max(don_i["Profil"]) == 1:
                don_i = goper.XY_Nan_completion_solo(don_i)
            else:
                don_i = goper.XY_Nan_completion(don_i)
        # Si la prospection est continue (détection ou précisé par l'utilisateur),
        # on cherche à construire des pseudo-profils
        if pseudo_prof or max(don_i["Profil"]) == 1:
            # Sélection dynamique
            if l_p == None:
                don_i = goper.detect_pseudoprof(don_i,"X_int","Y_int",l_p=None,plot=True)
                correct = False
                while correct == False:
                    gutils.input_mess(["Is profile detection correct ?","","y : Yes",
                                      "n : Retry without any segment (by default)",
                                      "Else, do a list of point coordinates to define a set of segments."
                                      "Exemple : \"50,20 82.5,0 100,-15\" initialize two segments"])
                    inp = input()
                    try:
                        if inp == "y":
                            correct = True
                        elif inp == "n":
                            don_i = goper.detect_pseudoprof(don_i,"X_int","Y_int",l_p=None,plot=True)
                        else:
                            pts = re.split(r"[ ]+",inp)
                            vect = [[float(c) for c in re.split(r",",pt)] for pt in pts]
                            if len(vect) < 2:
                                warnings.warn("Need at least 2 points.")
                            else:
                                don_i = goper.detect_pseudoprof(don_i,"X_int","Y_int",l_p=vect,plot=True)
                    except:
                        warnings.warn("Invalid answer.")
            # Si on veut juste prendre la droite de régression comme référence
            elif l_p == []:
                don_i = goper.detect_pseudoprof(don_i,"X_int","Y_int",l_p=None,plot=False)
            # Si on prend une liste de segments
            else:
                don_i = goper.detect_pseudoprof(don_i,"X_int","Y_int",l_p=l_p,plot=False)
            plt.close('all')
        # Séparation base/profil en fonction des colonnes "Base" et "Profil". La base peut être vide
        don_base,don_mes = goper.sep_BP(don_i)
        
        # Nom des colonnes de données
        nc_data = don_raw.columns[col_z]
        
        # Synthèse de chaque base en une seule ligne
        if not don_base.empty:
            d_nf,d_bp,d_t,d_min = emioper.synth_base(don_base,nc_data,CMDmini=(max(app_data["TR"])<2))
            don_base = d_min.transpose()
            don_base["File_id"] = d_nf
            don_base["Seconds"] = d_t
            don_base["B+P"] = d_bp
            don_base["Base"] = d_t.index+1
            don_base["Profil"] = 0
            
        # Traitements individuels : on prend un fichier à la fois
        for i in range(nb_file):
            i_fich_mes = don_mes[don_mes["File_id"] == i+1]
            i_fich_base = don_base[don_base["File_id"] == i+1]
            
            # Régression des profils pas droits (toujours dynamique)
            if regr:
                # Cas général
                if l_r == None:
                    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(CONFIG.fig_height,CONFIG.fig_height))
                    ax.plot(i_fich_mes["X_int"],i_fich_mes["Y_int"],'+r')
                    ax.plot(i_fich_mes[i_fich_mes["Profil"] == i_fich_mes.iloc[0]["Profil"]]["X_int"],
                            i_fich_mes["Y_int"][i_fich_mes["Profil"] == i_fich_mes.iloc[0]["Profil"]],'+b')
                    ax.set_xlabel(n_col_X)
                    ax.set_ylabel(n_col_Y)
                    ax.set_aspect('equal')
                    ax.set_title(file_list[i])
                    plt.show(block=False)
                    # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser 
                    # pour accélérer la vitesse de l'input
                    plt.pause(CONFIG.fig_render_time)
                    
                    correct = False
                    while correct == False:
                        gutils.input_mess(["File {} : regression ?".format(file_list[i]),
                                          "","y : Yes (all profiles)","k >= 0 : Yes, from profile k",
                                    "k < 0 : Yes, up to profile -k (-1 is last profil)",
                                    "n : No","","The first profil is shown in blue."])
                        inp = input()
                        try:
                            if inp == "n":
                                pass
                            elif inp == "y":
                                i_fich_mes = goper.pts_rectif(i_fich_mes)
                            elif int(inp) >= 0:
                                i_fich_mes = goper.pts_rectif(i_fich_mes,ind_deb=int(inp))
                            else:
                                i_fich_mes = goper.pts_rectif(i_fich_mes,ind_fin=int(inp))
                            correct = True
                        except ValueError:
                            warnings.warn("Invalid answer.")
                        except IndexError:
                            warnings.warn("Profile {} does not exist.".format(inp))
                    
                    plt.close(fig)
                    
                # Si on a listé les réponses au préalable
                else:
                    if len(l_r) != nb_file:
                        raise ValueError("Lengths of 'l_r' and 'file_list' do not match \
                                         ({} and {}).".format(len(l_r),nb_file))
                    try:
                        if l_r[i] == "n":
                            pass
                        elif l_r[i] == "y":
                            i_fich_mes = goper.pts_rectif(i_fich_mes)
                        elif int(l_r[i]) >= 0:
                            i_fich_mes = goper.pts_rectif(i_fich_mes,ind_deb=int(l_r[i]))
                        else:
                            i_fich_mes = goper.pts_rectif(i_fich_mes,ind_fin=int(l_r[i]))
                    except ValueError:
                        raise ValueError("'l_r' = {} : Invalid answer.".format(l_r[i]))
                    except IndexError:
                        raise ValueError("'l_r' = {} : Profile does not exist.".format(l_r[i]))
            
            # Étalonnage par base (pas de manuel depuis cette fonction)
            if corr_base:
                if not i_fich_base.empty:
                    i_fich_mes = evol_profiles_solo(i_fich_mes,i_fich_base,file_list[i],
                                                   col_z,app_data["nb_channels"],verif=False)
                else:
                    warnings.warn("No base in file '{}', thus no correction.".format(file_list[i]))
            
            # Décalage des positions par voies
            i_fich_mes = goper.dec_channels(i_fich_mes,ncx,ncy,app_data["nb_channels"],
                                         app_data["TR_l"],app_data["TR_t"],app_data["GPS_dec"])
            
            ls_mes.append(i_fich_mes)
            ls_base.append(i_fich_base)
            
            # Résultat enregistré en .dat (option)
            if in_file:
                i_fich_mes.to_csv(file_list[i][:-4]+"_init_P.dat", index=False, sep=sep) 
                if not i_fich_base.empty:
                    i_fich_base.to_csv(file_list[i][:-4]+"_init_B.dat", index=False, sep=sep)
    # Nom des colonnes de données, si tous les fichiers sont interpolés
    else:
        nc_data = ls_pd_done_before[0].columns[col_z]
    
    # Plot du résultat, en séparant chaque voie
    if plot:
        final_df = pd.concat(ls_mes)
        for e in range(app_data["nb_channels"]):
            fig,ax=plt.subplots(nrows=1,ncols=nb_res,figsize=(CONFIG.fig_width,CONFIG.fig_height))
            X = final_df[ncx[e]]
            Y = final_df[ncy[e]]
            for r in range(nb_res):
                n = e*nb_res + r
                Z = final_df[nc_data[n]]
                Q5,Q95 = Z.quantile([0.05,0.95])
                col = ax[r].scatter(X,Y,marker='s',c=Z,cmap='cividis',s=6,vmin=Q5,vmax=Q95)
                plt.colorbar(col,ax=ax[r],shrink=0.7)
                ax[r].title.set_text(nc_data[e*nb_res+r])
                ax[r].set_xlabel(ncx[e])
                ax[r].set_ylabel(ncy[e])
                ax[r].set_aspect('equal')
            plt.show(block=False)
            # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser 
            # pour accélérer la vitesse de l'input
            plt.pause(CONFIG.fig_render_time)
    
    # Sortie des dataframes (option)
    if not in_file:
        # Pour les besoins de la fonction 'main' (traitement général)
        if full_infos:
            return ls_base, ls_mes, ncx, ncy, nc_data, nb_res, ls_pd_done_before
        # Sortie classique
        else:
            return ls_base, ls_mes


def evol_profiles(file_prof_list,file_base_list,col_z,sep='\t',replace=False,
                 output_file_list=None,nb_channels=1,diff=True,base_adjust=True,
                 man_adjust=False,l_m=None,line=False,plot=False,in_file=False):
    """
    Main function for profile calibration from bases.\n
    See ``evol_profiles_solo`` for more infos.
    
    Notes
    -----
    Each file is managed separately.\n
    If ``base_adjust = False``, ignore ``file_base_list``.
    
    Parameters
    ----------
    file_prof_list : (list of str) or (list of dataframe)
        List of profiles files or loaded dataframes to process.
    file_base_list : (list of str) or (list of dataframe)
        List of bases files or loaded dataframes to process, ordered as ``file_prof_list``.
    col_z : list of int
        Index of every Z coordinates columns (actual data).
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` replace : bool, default : ``False``
        If the previous file is overwritten.
    ``[opt]`` output_file_list : ``None`` or list of str, default : ``None``
        List of output files names, ordered as ``file_prof_list``,
        otherwise add the suffix ``"_ep"``.
    ``[opt]`` nb_channels : int, default : ``1``
        Number of X and Y columns. The number of coils.
    ``[opt]`` diff : bool, default : ``True``
        Define which adjustment method (difference or ratio) is used.
    ``[opt]`` base_adjust : bool, default : ``True``
        Enables the first step.
    ``[opt]`` man_adjust : bool, default : ``False``
        Enables the second step.
    ``[opt]`` l_m : ``None`` or list of list of str, default : ``None``
        List of decisions (strings) for ``man_adjust``, orderd by 'file_prof_list'.
        If ``None``, enables a choice procedure.
    ``[opt]`` line : bool, default : ``False``
        Shows lines between profiles. Makes the visualization easier.
    ``[opt]`` plot : bool, default : ``False``
        Enables plotting.
    ``[opt]`` in_file : bool, default : ``False``
        If ``True``, save result in a file. If ``False``, return the dataframe.
    
    Returns
    -------
    * ``in_file = True``
        none, but save output dataframe in a .dat
    * ``in_file = False``
        file_list : dataframe
            Output dataframe list.

    Raises
    ------
    * ``file_prof_list``, ``file_base_list`` and/or ``output_file_list`` are different sizes.
    * ``man_adjust = True`` and ``l_m`` is in wrong format.
    
    See also
    --------
    ``evol_profiles_solo, check_time_date``
    """
    # Conversion en liste si 'file_prof_list' ne l'est pas
    if not isinstance(file_prof_list,list):
        file_prof_list = [file_prof_list]
    # Conversion en liste si 'file_base_list' ne l'est pas
    if not isinstance(file_base_list,list):
        file_base_list = [file_base_list]
    
    # Vérifier que les données sont cohérentes entre elles
    if base_adjust and len(file_prof_list) != len(file_base_list):
        raise ValueError("Lengths of 'file_prof_list' and 'file_base_list' do not match \
                         ({} and {}).".format(len(file_prof_list),len(file_base_list)))
    if not replace and output_file_list != None and len(file_prof_list) != len(output_file_list):
        raise ValueError("Lengths of 'file_prof_list' and 'output_file_list' do not match \
                         ({} and {}).".format(len(file_prof_list),len(output_file_list)))
    
    # Vérifier que 'l_m' est de la bonne taille
    if man_adjust and l_m != None:
        if len(l_m) != len(file_prof_list):
            raise ValueError("'l_m' must be of length {}, not {}".format(len(file_prof_list),len(l_m)))
    
    # Pour chaque fichier/dataframe
    res_list = []
    for i,file in enumerate(file_prof_list):
        if isinstance(file,str):
            # Chargement des données profils
            data_prof = gutils.check_time_date(file,sep)
            label = file
        else:
            data_prof = file
            label = str(i+1)
        if base_adjust:
            if isinstance(file_base_list[i],str):
                # Chargement des données bases
                data_base = gutils.check_time_date(file_base_list[i],sep)
            else:
                data_base = file_base_list[i]
        else:
            data_base = pd.DataFrame()
        
        # Gestion du manuel auto
        if man_adjust and l_m != None:
            l_m_ = l_m[i]
        else:
            l_m_ = None
        
        # Procédure
        res = evol_profiles_solo(data_prof,data_base,label,col_z,nb_channels,diff,
                                base_adjust,man_adjust,l_m_,plot,line)
        
        if not in_file:
            res_list.append(res)
        # Résultat enregistré en .dat (option)
        else:
            if replace:
                res.to_csv(file[i], index=False, sep=sep)
            elif output_file_list == None:
                res.to_csv(file[i][:-4]+"_ep.dat", index=False, sep=sep)
            else:
                res.to_csv(output_file_list[i], index=False, sep=sep)
    # Sortie des dataframes (option)
    if not in_file:
        return res_list


def evol_profiles_solo(prof,bas,nom_fich,col_z,nb_channels,diff=True,base_adjust=True,
                      man_adjust=False,l_m=None,plot=False,line=False):
    """
    Given a profile database and an associated base database, perform profile calibration
    by alignment of bases (bases are supposed to give the same value each time).\n
    The operation is performed by difference, but it is also possible to perform it
    by multiplication (ratio).\n
    It is possible to request the rectification of profile blocks if other imperfections
    are visible, using ``man_adjust = True``.\n
    If you only want to perform this operation, you can disable the first step
    using the ``base_adjust = False``.
    
    Notes
    -----
    If used as a standalone, plots every step.\n
    If ``base_adjust = False``, ignore ``bas``.\n
    If both ``man_adjust`` and ``base_adjust`` are set to false, nothing happens.
    
    Parameters
    ----------
    prof : dataframe
        Profile dataframe.
    bas : dataframe
        Base dataframe.
    nom_fich : str
        Profile file name. Used in plots.
    col_z : list of int
        Index of every Z coordinates columns (actual data).
    nb_channels : int
        Number of X and Y columns. The number of coils.
    ``[opt]`` diff : bool, default : ``True``
        Define which adjustment method (difference or ratio) is used.
    ``[opt]`` base_adjust : bool, default : ``True``
        Enables the first step.
    ``[opt]`` man_adjust : bool, default : ``False``
        Enables the second step.
    ``[opt]`` l_m : ``None`` or list of str, default : ``None``
        List of decisions (strings) for ``man_adjust``.
        If ``None``, enables a choice procedure.
    ``[opt]`` plot : bool, default : ``False``
        Enables plotting.
    ``[opt]`` line : bool, default : ``False``
        Shows lines between profiles. Makes the visualization easier.
    
    Returns
    -------
    don : dataframe
        Updated profile dataframe
    
    Raises
    ------
    * Profile dataframe is not interpolated.
    * Base dataframe does not contains a ``"Base"`` column.
    * ``man_adjust = True`` and ``l_m`` is in wrong format.
    
    See also
    --------
    ``evol_profiles, check_time_date``
    """
    global GUI_VAR_LIST
    
    # Tentative désespérée de régler un soucis, mais pas sûr que ça soit nécessaire
    don = prof.copy()
    
    # Options du plot
    color = ["blue","green","orange","magenta","red","cyan","black","yellow"]
    if line:
        mrk = 'x-'
    else:
        mrk = 'x'
    
    # Si la colonne 'Profil' n'existe pas, le fichier n'est pas interpolé ou invalide
    try:
        prof_deb = don['Profil'].iat[0]
        prof_fin = don['Profil'].iat[-1]
    except KeyError:
        raise KeyError("Profile dataframe is not interpolated, see 'init_process'.")
    prof_l = prof_fin-prof_deb+1
    
    col_names = don.columns[col_z]
    nb_data = len(col_z)
    nb_res = max(1, nb_data//nb_channels)
    
    prof_med = np.array([[0.0]*prof_l]*nb_data)
    prof_bp = []
    index_list = []
    
    # Profil : calcul de la médiane + on note la correspondance "Profil"/"B+P"
    # et les index de début de profil
    for i in range(prof_l):
        prof = don[don["Profil"] == i+prof_deb]
        prof_bp.append(prof['B+P'].iat[0])
        index_list.append(prof.index[0])
        for j in range(nb_data):
            prof_med[j,i] = prof[col_names[j]].median()
    index_list.append(None)
    
    if base_adjust:
        # Si la colonne 'Base' n'existe pas, le fichier n'est pas interpolé ou invalide
        try:
            base_deb = bas['Base'].iat[0]
            base_fin = bas['Base'].iat[-1]
        except KeyError:
            raise KeyError("Base dataframe is not interpolated nor fused correctly, \
                           see 'init_process' or 'fuse_bases'.")
        base_l = base_fin-base_deb+1
        
        base_med = np.array([[0.0]*base_l]*nb_data)
        base_bp = []
        
        # Profil : calcul de la médiane + on note la correspondance "Base"/"B+P"
        for i in range(base_l):
            base = bas[bas["Base"] == i+base_deb]
            base_bp.append(base['B+P'].iat[0])
            for j in range(nb_data):
                base_med[j,i] = base[col_names[j]].median()
        
        # Plot des médianes des données initiales
        if plot:
            fig,ax = plt.subplots(nrows=1,ncols=nb_res,
                                  figsize=(nb_res*CONFIG.fig_width//2,CONFIG.fig_height),
                                  squeeze=False)
            if diff:
                for j in range(nb_data):
                    ax[0][j%nb_res].plot(prof_bp,(prof_med[j]-prof_med[j,0]),mrk,
                                         label=col_names[j]+" (profile)",color=color[j//nb_res])
                    ax[0][j%nb_res].plot(base_bp,(base_med[j]-base_med[j,0]),'o--',
                                         label=col_names[j]+" (base)",color=color[j//nb_res])
                    ax[0][j%nb_res].set_xlabel("Profile")
                    ax[0][j%nb_res].set_ylabel("Difference with first")
                    ax[0][j%nb_res].grid(axis="x")
                    ax[0][j%nb_res].legend()
            else:
                for j in range(nb_data):
                    ax[0][j%nb_res].plot(prof_bp,prof_med[j]/max(prof_med[j]),mrk,
                                         label=col_names[j]+" (profile)",color=color[j//nb_res])
                    ax[0][j%nb_res].plot(base_bp,base_med[j]/max(base_med[j]),'o--',
                                         label=col_names[j]+" (base)",color=color[j//nb_res])
                    ax[0][j%nb_res].set_xlabel("Profile")
                    ax[0][j%nb_res].set_ylabel("Proportion with first")
                    ax[0][j%nb_res].grid(axis="x")
                    ax[0][j%nb_res].legend()
            fig.suptitle(nom_fich+" (initial data)")
            plt.show(block=False)
            # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser 
            # pour accélérer la vitesse de l'input
            plt.pause(CONFIG.fig_render_time)
        
        # Retrait de la valeur de la base pour le signal en phase
        in_phase_cols = [ic for ic,c in enumerate(col_names) if "Inph" in c]
        for Inph in in_phase_cols:
            don.loc[:, col_names[Inph]] = (don.loc[:, col_names[Inph]] - base_med[Inph,0])\
                .round(CONFIG.prec_data)
        # # Retrait de la valeur de la base pour le signal en quadrature
        # cond_cols = [ic for ic,c in enumerate(col_names) if "Cond" in c]
        # for Cond in cond_cols:
        #     don.loc[:, col_names[Cond]] = (don.loc[:, col_names[Cond]] - base_med[Cond,0])\
        #        .round(CONFIG.prec_data)
        
        #Ajustement par base
        prev_base_fact = [[base_med[j,k] - base_med[j,0] for k in range(len(base_bp))] for j in range(nb_data)]
        for i in range(prof_l):
            prof = don[don["Profil"] == i+prof_deb]
            r = prof["B+P"].iat[0]
            k = 0
            while k < base_l and base_bp[k] < r:
                k = k+1
            
            # Gestion des cas début/fin
            av = k-1
            ap = k
            if k == 0:
                av = 0
                fact = 0
            elif k == base_l:
                ap = base_l-1
                fact = 1
            else:
                fact = (r-base_bp[av])/(base_bp[ap]-base_bp[av])
            for j in range(nb_data):
                # Affectation de la valeur après ajustement
                if diff:
                    aj = fact*(base_med[j,ap]-base_med[j,av])
                    new_val = prof[col_names[j]] - aj - prev_base_fact[j][av]
                else:
                    aj = (fact*base_med[j,ap] + (1-fact)*base_med[j,av])
                    new_val = prof[col_names[j]]/aj - prev_base_fact[j][av]
                if index_list[i+1] != None:
                    temp = don.loc[index_list[i+1]][col_names[j]]
                don.loc[index_list[i]:index_list[i+1], col_names[j]] = new_val.round(CONFIG.prec_data)
                if index_list[i+1] != None:
                    don.loc[index_list[i+1], col_names[j]] = temp
            #print(i,i+prof_deb,av,ap,prev_base_first,prev_base_fact,fact)
        
        for i in range(prof_l):
            prof = don[don["Profil"] == i+prof_deb]
            for j in range(nb_data):
                prof_med[j,i] = prof[col_names[j]].median()
    
    if plot or man_adjust:
        cpt = 0 # Uniquement pour 'l_m'
        correct = False
        while correct == False:
            # Plot des médianes des données après ajusement par base (si il a eu lieu)
            if l_m == None:
                fig,ax = plt.subplots(nrows=1,ncols=nb_res,
                                      figsize=(nb_res*CONFIG.fig_width//2,CONFIG.fig_height),
                                      squeeze=False)
                if diff:
                    for j in range(nb_data):
                        ax[0][j%nb_res].plot(prof_bp,(prof_med[j]-prof_med[j,0]),mrk,
                                             label=col_names[j]+" (profile)",color=color[int(j/nb_res)])
                        ax[0][j%nb_res].set_xlabel("Profile")
                        ax[0][j%nb_res].set_ylabel("Difference with first")
                        ax[0][j%nb_res].grid(axis="x")
                        ax[0][j%nb_res].legend()
        
                else:
                    for j in range(nb_data):
                        ax[0][j%nb_res].plot(prof_bp,prof_med[j]/max(prof_med[j]),mrk,
                                             label=col_names[j]+" (profile)",color=color[int(j/nb_res)])
                        ax[0][j%nb_res].set_xlabel("Profile")
                        ax[0][j%nb_res].set_ylabel("Proportion with first")
                        ax[0][j%nb_res].grid(axis="x")
                        ax[0][j%nb_res].legend()
                if base_adjust:
                    fig.suptitle(nom_fich+" (adjusted data)")
                else:
                    fig.suptitle(nom_fich)
                plt.show(block=False)
                # À augmenter si la figure ne s'affiche pas, sinon on pourra le baisser
                # pour accélérer la vitesse de l'input
                plt.pause(CONFIG.fig_render_time)
            
            # Ajustement manuel
            if man_adjust:
                try:
                    if l_m != None:
                        try:
                            inp = l_m[cpt]
                            cpt += 1
                        except IndexError:
                            inp = "n"
                    else:
                        gutils.input_mess(["Select first the bloc of profile to correct, \
                                          then the columns in which apply the regression.",
                                         "This procedure is launched until 'n'.",
                                         "","a-b x y z: From profile a to b (included), \
                                            on columns x,y and z","n : No (end procedure)"])
                        inp = input()
                        print(inp)
                    if inp == "n":
                        correct = True
                    else:
                        res = re.split(r"[ ]+",inp)
                        id_prof = re.split(r"-",res[0])
                        first = don[don["B+P"] == int(id_prof[0])]["Profil"].iat[0]-prof_deb
                        last = don[don["B+P"] == int(id_prof[1])]["Profil"].iat[0]-prof_deb
                        column_to_do = []
                        for r in res[1:]:
                            column_to_do.append(int(r)-1)
                        new_med = []
                        
                        # Gestion des cas début/fin
                        av = first
                        ap = last
                        if first == 0:
                            av = last
                            if last == 0:
                                av = first+1
                            if last == prof_fin-prof_deb:
                                av = first+1
                                ap = last-1
                        elif last == prof_fin-prof_deb:
                            ap = first  
                            if first == prof_fin-prof_deb:
                                ap = last-1
                        
                        # Relation linéaire pour la régression entre la borne gauche et droite
                        for j in range(nb_data):
                            new_med.append(np.linspace(prof_med[j,av-1],
                                                       prof_med[j,ap+1],(last-first)+3))
                        for i in range(first,last+1):
                            prof = don[(don["Profil"] == i+prof_deb)]
                            if prof.empty: # Base
                                continue
                            r = prof["B+P"].iat[0]
                            k = 0
                            while k < prof_l and prof_bp[k] < r:
                                k = k+1
                            for j in column_to_do:
                                # Affectation de la valeur après ajustement
                                if diff:
                                    new_val = prof[col_names[j]] + (new_med[j][i-first+1]-prof_med[j,k])
                                else:
                                    new_val = prof[col_names[j]] * (new_med[j][i-first+1]/prof_med[j,k])
                                if index_list[k+1] != None:
                                    temp = don.loc[index_list[k+1]][col_names[j]]
                                don.loc[index_list[k]:index_list[k+1], 
                                        col_names[j]] = new_val.round(CONFIG.prec_data)
                                if index_list[k+1] != None:
                                    don.loc[index_list[k+1], col_names[j]] = temp
                        
                        # Nouveau calcul des médianes des profils (pour un autre ajustement)
                        for i in range(prof_l):
                            prof = don[don["Profil"] == i+prof_deb]
                            for j in range(nb_data):
                                prof_med[j,i] = prof[col_names[j]].median()
                except ValueError:
                    if l_m == None:
                        warnings.warn("Invalid answer.")
                    else:
                        raise ValueError("'l_m' = {} : Invalid answer.".format(l_m[cpt-1]))
                except IndexError:
                    if l_m == None:
                        warnings.warn("Profile {} does not exist.".format(id_prof))
                    else:
                        raise ValueError("'l_m' = {} : Profile does not exist.".format(l_m[cpt-1]))
                plt.close()
            else:
                correct = True
    
    # Affichage final si on a automatisé l'ajustement manuel
    if man_adjust and plot and l_m != None:
        fig,ax = plt.subplots(nrows=1,ncols=nb_res,
                              figsize=(nb_res*CONFIG.fig_width//2,CONFIG.fig_height),
                              squeeze=False)
        if diff:
            for j in range(nb_data):
                ax[0][j%nb_res].plot(prof_bp,(prof_med[j]-prof_med[j,0]),mrk,
                                     label=col_names[j]+" (profile)",color=color[int(j/nb_res)])
                ax[0][j%nb_res].set_xlabel("Profile")
                ax[0][j%nb_res].set_ylabel("Difference with first")
                ax[0][j%nb_res].grid(axis="x")
                ax[0][j%nb_res].legend()

        else:
            for j in range(nb_data):
                ax[0][j%nb_res].plot(prof_bp,prof_med[j]/max(prof_med[j]),mrk,
                                     label=col_names[j]+" (profile)",color=color[int(j/nb_res)])
                ax[0][j%nb_res].set_xlabel("Profile")
                ax[0][j%nb_res].set_ylabel("Proportion with first")
                ax[0][j%nb_res].grid(axis="x")
                ax[0][j%nb_res].legend()
    
    # Sortie du dataframe ajusté
    return don.copy()


def calibration(uid,col_ph,col_qu,file_list=None,eff_sigma=True,show_steps=True,light=True,
                sep='\t',output_file_list=None,in_file=False):
    """
    Given two arrays ``X`` and ``Y``, compute the coefficients of the chosen regression.\n
    To be used in the context of finding a formula for a physical relation.
    
    Parameters
    ----------
    uid : int or dict
        Device's ``"app_id"`` value, or the loaded dictionary of device's parameters.
    col_ph : list of int
        Index of every ``Inphase`` columns.
    col_ph : list of int
        Index of every ``Conductivity`` columns.
    ``[opt]`` file_list : ``None`` or (list of) str or (list of) dataframe, default : ``None``
        List of files to process.
    ``[opt]`` eff_sigma : str, default : ``True``
        If we remove the computed effect of sigma for inphase signal.
    ``[opt]`` show_steps : bool, default : ``True``
        Prints an increment every 250 points, as well as the current coil.
    ``[opt]`` light : bool, default : ``False``
        Only keep final columns (sigma and kappa).
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` output_file_list : ``None`` or list of str, default : ``None``
        List of output files names, ordered as ``file_list``, 
        otherwise add the suffix ``"_calibr"``.
    ``[opt]`` in_file : bool, default : ``False``
        If ``True``, save result in a file. If ``False``, return the dataframe.
    
    Returns
    -------
    * ``in_file = True``
        none, but save output dataframe in a .dat
    * ``in_file = False``
        df : dataframe
            Output dataframe.
    
    Notes
    -----
    [DEV] The ``conv`` parameter may be removed if one of the two procedure 
    is deemed better in all cases.\n
    TODO : Still in progress.
        
    Raises
    ------
    * File not found.
    * ``file_list`` and ``output_file_list`` are different sizes.
    * Fortran executable fails.
    * Unknown OS.
    
    See also
    --------
    ``add_coeff_to_json, _file_constab, coeffs_relation, find_device, true_file_list``
    """
    # Répertoire de travail de l'utilisateur
    user_cwd = os.getcwd()
    
    # Conversion en liste si 'file_list' ne l'est pas
    try:
        if file_list != None and not isinstance(file_list,list):
            file_list = [file_list]
        file_list = gutils.true_file_list(file_list)
    # Type dataframe
    except ValueError:
        file_list = [file_list]
    if output_file_list != None and len(file_list) != len(output_file_list):
        raise ValueError("Lengths of 'file_list' and 'output_file_list' do not match \
                         ({} and {}).".format(len(file_list),len(output_file_list)))
    
    # Chargement de l'appareil si 'uid' est l'indice dans la base 'Appareils.json'
    if isinstance(uid,int):
        app_data = emijson.find_device(uid)
    else:
        app_data = uid
    
    # Chargement des constantes
    const_dict = emijson.add_coeff_to_json(app_data)
    #print(const_dict)
    sigma_a_ph = const_dict["sigma_a_ph"]
    sigma_a_qu = const_dict["sigma_a_qu"]
    # Inversion de signe des constantes, fonction de la configuration
    if app_data["config"] in ["HCP"]:
        for e in range(app_data["nb_channels"]):
            sigma_a_ph[e] = [-c for c in sigma_a_ph[e]]
    if True:
        for e in range(app_data["nb_channels"]):
            sigma_a_qu[e] = [-c for c in sigma_a_qu[e]]
    
    # Nom de fichiers
    cfg_file = "_config_.cfg"
    fortran_exe = "terrainhom.exe"
    fortran_linux = "terrainhom.out"
    
    res_list = []
    for ic, file in enumerate(file_list):
        os.chdir(user_cwd)
        if isinstance(file,str):
            try:
                # Chargement des données
                df_ = pd.read_csv(file, sep=sep)
                # Si on a envie de prendre moins de points
                df = df_[::]
            except FileNotFoundError:
                raise FileNotFoundError('File "{}" not found.'.format(file))
        else:
            df = file
        
        # On traîte chaque voie séparément
        os.chdir(CONFIG.emi_fortran_path)
        for e in range(app_data["nb_channels"]):
            ncph = df.columns[col_ph[e]]
            ncqu = df.columns[col_qu[e]]
            
            # Colonnes des signaux après application des coeffs constructeurs
            df[ncph+"_aj"] = df[ncph]*app_data["coeff_c_ph"][e] # unité en ppt => ppt
            df[ncqu+"_aj"] = df[ncqu]*app_data["coeff_c_qu"][e] # unité en mS/m => ppt
            
            # Nouvelles colonnes 
            df["sigma_"+str(e+1)] = 0
            df["Kph_a_ph_"+str(e+1)] = 0
            df["Kph_"+str(e+1)] = 0
            df["eff_sig_"+str(e+1)] = 0
        
            v = 100
            print("e = ",e)
            for index,row in df.iterrows():
                qu = row[ncqu+"_aj"]/1000 # unité en ppt => SI
                
                # Sigma du point
                sigma_c = (qu*sigma_a_qu[e][1] + qu**2*sigma_a_qu[e][2] + qu**3*sigma_a_qu[e][3])
                
                # Cas en dehors de l'intervalle valide (objet métallique ?)
                if sigma_c <= 0:
                    df.loc[index,"eff_sig_"+str(e+1)] = np.nan
                    df.loc[index,"sigma_"+str(e+1)] = np.nan
                    df.loc[index,"Kph_a_ph_"+str(e+1)] = np.nan
                    df.loc[index,"Kph_"+str(e+1)] = np.nan
                    if index%250 == 0:
                        print(index)
                    continue
                
                # Création du fichier appelé par le Fortran
                emijson._file_constab(app_data,cfg_file,e,variation=v,
                                      S_rau=1/sigma_c,S_eps_r=1,S_kph=0.1E-5,S_kqu=0.1E-7,
                                      F_rau=None,F_eps_r=None,F_kph=0.01,F_kqu=None)
                
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
                # print(index)
                don = pd.read_csv(cfg_file[:-4]+".dat",sep='\s+',header=None)
                #print(don)
                
                # Calcul de l'effet de sigma sur la phase
                if eff_sigma:
                    eff_sig = (sigma_c*sigma_a_ph[e][1] + sigma_c**2*sigma_a_ph[e][2] + sigma_c**3*sigma_a_ph[e][3]) # unité en SI => SI
                else:
                    eff_sig = 0
                
                # Inversion du signe de la contribution en fonction de la configuration
                if app_data["config"] in ["VCP"]:
                    eff_sig = eff_sig
                # Signal en phase (tableau) sans l'effet de sigma
                HsHp_ph = np.array(don.iloc[:,4])
                # Kappa en phase (tableau)
                Kph_col = np.array(don.iloc[:,2])
                # Coefficient du signal en phase à kappa phase
                Kph_a_ph = goper.coeffs_relation(HsHp_ph,Kph_col,m_type="linear")[1]
                #print(Kph_a_ph)
                
                # Calcul de kappa phase
                Kph = (row[ncph+"_aj"]/1000 - eff_sig)*(-Kph_a_ph) # unité en SI => SI
                #print(Kph)
                
                # Insertion des valeurs dans le dataframe
                df.loc[index,"eff_sig_"+str(e+1)] = eff_sig
                df.loc[index,"sigma_"+str(e+1)] = sigma_c
                df.loc[index,"Kph_a_ph_"+str(e+1)] = Kph_a_ph
                df.loc[index,"Kph_"+str(e+1)] = Kph
                if index%250 == 0:
                    print(index)
            
            # Arrondi des colonnes des signaux initiaux, si la lecture les a déformés
            df.loc[:,ncph] = df[ncph].round(CONFIG.prec_data)
            df.loc[:,ncqu] = df[ncqu].round(CONFIG.prec_data)
        
        os.chdir(user_cwd)
        # Sortie du dataframe (option)
        if not in_file:
            res_list.append(df)
        # Résultat enregistré en .dat (option)
        elif output_file_list == None:
            df.to_csv(file[:-4]+"_calibr.dat", index=False, sep=sep)
        else:
            df.to_csv(output_file_list[ic], index=False, sep=sep)

    if not in_file:
        return res_list