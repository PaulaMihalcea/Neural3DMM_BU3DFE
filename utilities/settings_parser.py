from configparser import ConfigParser

cfg_path = './settings/setup_file'


# SETUP FILE GETTER
def get_setup_file():

    f = ConfigParser()
    f.read(cfg_path)  # Parse the setup.cfg file to retrieve settings

    setup_file = f.get('Setup file', 'setup_file')

    return setup_file


# SETTINGS GETTER
def get_settings(section):
    f = ConfigParser()

    cfg_path = get_setup_file()

    f.read(cfg_path)  # Parse the setup.cfg file to retrieve settings

    settings = {}

    for key in f[section]:
        settings[key] = f[section][key]

    return settings


# SETUP FILE SETTER
def set_setup_file(value):
    f = ConfigParser()
    f.read(cfg_path)

    f.set('Setup file', 'setup_file', value)

    with open(cfg_path, 'w') as cfg_file:
        f.write(cfg_file)

    return
