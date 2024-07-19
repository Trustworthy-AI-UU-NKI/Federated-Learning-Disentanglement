class Config():
    def __init__(self, config_parser_object, cfg_name):
        self.sections = []
        self.__configname__ = cfg_name

        for section in config_parser_object.keys():
            if section != "DEFAULT":    # skip DEFAULT section which is always created by configparser
                self.sections.append(section)
                setattr(self, section, ConfigSection(config_parser_object[section]))

    def __repr__(self):
        str_repr = ""
        for section in self.sections:
            str_repr += f"\n[{section}]{getattr(self, section)}\n"
        return str_repr

    def get_name(self):
        return self.__configname__

    def get_dict(self):
        full_d = {}
        for section in self.sections:
            full_d.update(getattr(self, section).get_dict())

        return full_d

class ConfigSection():
    def __init__(self, section):
        self.keys = []

        for key in section.keys():
            self.keys.append(key)
            str_val = section[key]

            if str_val.isdigit():
                val = int(str_val)
            elif str_val.replace('.', '').isdigit():
                val = float(str_val)
            elif str_val == "None":
                val = None
            elif str_val == "True":
                val = True
            elif str_val == "False":
                val = False
            else:
                val = str_val

            setattr(self, key, val)

    def get_dict(self):
        d = {}
        for key in self.keys:
            d[key] = getattr(self, key)

        return d

    def __repr__(self):
        str_repr = ""
        for key in self.keys:
            str_repr += f"\n{key}: {getattr(self, key)}"
        return str_repr


def extract_data_config(config):
    data_config = {
        "batch_size": config.data.batch_size,
        "target_size": config.data.target_size,
        "out_center": config.data.out_center,
        "splits": config.data.splits,
        "seed": config.data.seed,
        "load_in_ram": config.data.load_in_ram,
        "load_names": config.data.load_names,
        "fold_nr": config.data.fold_nr
    }

    return data_config