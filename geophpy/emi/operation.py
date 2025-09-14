# -*- coding: utf-8 -*-
'''
   geophpy.emi.operation
   -------------------------

   DataSet Object electromagnetic operations routines.

   :copyright: Copyright 2014-2025 Lionel Darras, Philippe Marty, Quentin Vitale and contributors, see AUTHORS.
   :license: GNU GPL v3.
'''

import pandas as pd

import geophpy.core.operation as goper
from geophpy.core.utils import check_time_date
import geophpy.__config__ as CONFIG

def fuse_bases(file_B1,file_B2,file_prof,sep='\t',output_file=None,in_file=False):
    """
    Given two files containing bases from the same prospection, fuse them and add ``"B+P"`` and ``"Base"`` columns.\n
    To be used if bases have been taken separately.
    
    Notes
    -----
    ``file_B1`` is done before ``file_prof``, whereas ``file_B2`` is done after.\n
    ``file_prof`` is required in order to get the right value of ``"B+P"``.
    
    Parameters
    ----------
    file_B1 : str
        Base 1 file.
    file_B2 : str
        Base 2 file.
    file_prof : str
        Profile file corresponding to given bases.
    ``[opt]`` sep : str, default : ``'\\t'``
        Dataframe separator.
    ``[opt]`` output_file : ``None`` or str, default : ``None``
        Output file name, otherwise add the suffix ``"_B"`` to ``file_prof``.
    ``[opt]`` in_file : bool, default : ``False``
        If ``True``, save result in a file. If ``False``, return the dataframe.
    
    Returns
    -------
    * ``in_file = True``
        none, but save output dataframe in a .dat
    * ``in_file = False``
        base : dataframe
            Output base dataframe.
    
    Raises
    ------
    * File not found.
    * Wrong separator or columns not found.
    """
    # Chargement des données
    try:
        B1 = check_time_date(file_B1,sep)
    except FileNotFoundError:
        raise FileNotFoundError("File B1 '{}' not found.".format(file_B1))
    try:
        B2 = check_time_date(file_B2,sep)
    except FileNotFoundError:
        raise FileNotFoundError("File B2 '{}' not found.".format(file_B2))
    try:
        prof = check_time_date(file_prof,sep)
    except FileNotFoundError:
        raise FileNotFoundError("File of profiles '{}' not found.".format(file_prof))
    
    # Test interpolation
    try:
        B1["B+P"], B1["Base"], B2["B+P"], B2["Base"] = 0, 1, int(prof['B+P'].iat[-1])+1, 2
    except KeyError:
        raise KeyError("File '{}' is not interpolated or does not have '{}' as its separator.".format(file_prof,repr(sep)))
    
    # On suppose que les valeurs basses sont en lugne impair (à améliorer)
    base = pd.concat([B1,B2])
    base.reset_index(drop=True,inplace=True)
    
    base["File_id"] = 1
    d_nf,d_bp,d_t,d_min = synth_base(base,base.columns[[2,3,5,6,8,9]],True)
    base = d_min.transpose()
    base["File_id"] = d_nf
    base["Seconds"] = d_t
    base["B+P"] = d_bp
    base["Base"] = d_t.index+1
    base["Profil"] = 0
    
    # Sortie du dataframe (option)
    if not in_file:
        return base
    # Résultat enregistré en .dat (option)
    if output_file == None:
        base.to_csv(file_prof[:-4]+"_B.dat", index=False, sep=sep)
    else:
        base.to_csv(output_file, index=False, sep=sep)


def synth_base(don,nc_data,CMDmini=True):
    """
    Resume each base by one single line.\n
    It contains all data pointed by ``nc_data``, its profile/base index and its time.\n
    Does not return a single dataframe, is managed by ``init``.
    
    Parameters
    ----------
    don : dataframe
        Active base dataframe.
    nc_data : list of str
        Names of every conductivity and inphase columns, ordered two by two.
    ``[opt]`` CMDmini : bool, default : ``True``
        If bases were taken in the air (device's 0). 
    
    Returns
    -------
    pd_num_fich : pd.Series
        ``"File_id"`` column (file order).
    pd_bp : pd.Series
        ``"B+P"`` column (base + profile index).
    pd_tps : pd.Series
        ``"Seconds"`` column (base + profile index).
    * ``CMDmini = True``
        pd_inf : pd.Series
            Data columns for lowest values (device's 0).
    * ``CMDmini = False``
        pd_valmd : pd.Series
            Data columns for average values, since all points are on the ground.
    
    Notes
    -----
    Subfunction of ``init_process``.\n
    If ``CMDmini = False``, the result will not give enough information to remove
    the device's 0, although the variations between bases will stand.
    
    See also
    --------
    ``init_process, evol_profiles``
    """
    # Division des colonnes entre conductivité et inphase
    cond_data = nc_data[::2]
    inph_data = nc_data[1::2]
    
    # Vérification de format
    if not('Base' in don.columns) :
        raise KeyError("Call 'detect_base_pos' before this procedure.")
    if not('Seconds' in don.columns) :
        try:
            don['Seconds'] = goper.set_time(don)
        except KeyError:
            don['Seconds'] = -1
    
    # Calcul des quantiles pour diviser la base entre haut et bas
    # On utilise la conductivité pour faire la différence
    num_base=don['Base'].unique()
    num_base=num_base[num_base>0]    
    ls_num_fich,ls_bp,ls_tps,ls_val=[],[],[],[]
    for n_base in num_base :
        ind_c=don.index[don['Base']==n_base]
        tps_c=don.loc[ind_c,'Seconds'].median()
        Q5=don.loc[ind_c,cond_data].quantile(0.05)
        Q95=don.loc[ind_c,cond_data].quantile(0.95)
        valb_c=(Q95+Q5)/2.
        ls_num_fich.append(don.loc[ind_c[0],"File_id"])
        ls_bp.append(don.loc[ind_c[0],"B+P"])
        ls_tps.append(tps_c),ls_val.append(valb_c)
    
    # Association de la moyenne des valeurs en haut pour chaque base
    pd_valmd=pd.concat(ls_val,axis=1)
    ls_sup,ls_inf=[],[]
    for n_base in num_base :
        ind_c=don.index[don['Base']==n_base]
        seuil=pd_valmd[n_base-1]
        ls_s,ls_i=[],[]
        for ic,sc in enumerate(seuil) :
            dat_c=don.loc[ind_c,cond_data[ic]].copy()
            dat_i=don.loc[ind_c,inph_data[ic]].copy()
            ind1=dat_c.index[dat_c>sc]
            ind2=dat_c.index[dat_c<sc]
            ls_s.append(dat_c.loc[ind1].median())     
            ls_i.append(dat_c.loc[ind2].median())
            ls_s.append(dat_i.loc[ind1].median())     
            ls_i.append(dat_i.loc[ind2].median())
        
        
        
        ls_sup.append(pd.Series(ls_s))
        ls_inf.append(pd.Series(ls_i))
    
    # Mise au bon format
    pd_sup=pd.concat(ls_sup,axis=1).round(CONFIG.prec_data)
    pd_inf=pd.concat(ls_inf,axis=1).round(CONFIG.prec_data)
    pd_sup.index=nc_data
    pd_inf.index=nc_data 
    pd_med = pd_sup.add(pd_inf, fill_value=0)/2
    pd_num_fich=pd.Series(ls_num_fich)  
    pd_bp=pd.Series(ls_bp)     
    pd_tps=pd.Series(ls_tps).round(CONFIG.prec_data)
    
    # Si l'appareil a pu être soulevé ou non
    if CMDmini :
        return(pd_num_fich,pd_bp,pd_tps,pd_inf)
    else :
        return(pd_num_fich,pd_bp,pd_tps,pd_med)
