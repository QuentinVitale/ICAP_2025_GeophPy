# -*- coding: utf-8 -*-
'''
    geophpy.core.json
    ---------------------------

    General intern json file management and related Fortran scripts.

    :copyright: Copyright 2014-2025 L. Darras, P. Marty, Q. Vitale and contributors, see AUTHORS.
    :license: GNU GPL v3.

'''

import json

class JSON_Indent_Encoder(json.JSONEncoder):
    """ 
    Modify the JSON indent rule for readability (`Constantes.json`).
    """
    def iterencode(self, o, _one_shot=False):
        dict_lvl = 0
        list_lvl = 0
        for s in super(JSON_Indent_Encoder, self).iterencode(o, _one_shot=_one_shot):
            if s.startswith('{'):
                dict_lvl += 1
                s = s.replace('{', '{\n'+dict_lvl*"  ")
            elif s.startswith('}'):
                dict_lvl -= 1
                s = s.replace('}', '\n'+dict_lvl*"  "+'}')
            elif s.startswith('['):
                list_lvl += 1
            elif s.startswith(']'):
                list_lvl -= 1
            elif s.startswith(',') and list_lvl == 8:
                s = s.replace(',', ',\n'+dict_lvl*"  ")
            yield s