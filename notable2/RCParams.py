import os
from DataGetters import packETGetter

"""TODO put in actuall rc file and parse it"""

home = os.environ["HOME"]
if (xdg_config_home := os.environ["XDG_CONFIG_HOME"]) == '':
    if home != '':
        xdg_config_home = f'{home}/.config'


RCParams = dict(default_getter=packETGetter,
                default_eos_path=f'{home}/desert/simulations/EOS/DD2')
