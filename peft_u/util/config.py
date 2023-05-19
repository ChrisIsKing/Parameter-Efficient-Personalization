from os.path import join as os_join
from stefutil import *

config_dict = dict()

if __name__ == '__main__':
    import json
    from project_paths import BASE_PATH, PROJ_DIR, PKG_NM

    mic.output_width = 256

    fl_nm = 'config.json'
    mic(config_dict)
    open(fl_nm, 'a').close()  # Create file in OS
    with open(os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', fl_nm), 'w') as f:
        json.dump(config_dict, f, indent=4)
