import configparser
import ast


class Config:
    def __init__(self, path):
        self.cfg = configparser.ConfigParser()
        self.cfg.read(path)

    def get(self, section, option, star=None, raw=False, vars=None, fallback=configparser._UNSET):
        value = self.cfg.get(section, option, raw=raw, vars=vars, fallback=fallback)
        try:
            return ast.literal_eval(value)
        except Exception:
            return value

