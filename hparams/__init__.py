import json
from attrdict import AttrDict
from typing import Any
import yaml

def json_to_dict(json_file:str) -> dict:
    with open(json_file,'r') as f:
        json_loaded = json.load(f)
    return json_loaded

def yaml_to_dict(yaml_file:str) -> dict:
    with open(yaml_file,'r') as f:
        loaded = yaml.safe_load(f)
    return loaded

def dict_to_attr(dictionary:dict) ->Any:
    return AttrDict(dictionary)

def json_to_attr(json_file:str) -> Any:
    d = json_to_dict(json_file)
    attr_cls = dict_to_attr(d)
    return attr_cls

def yaml_to_attr(yaml_file:str) -> Any:
    d = yaml_to_dict(yaml_file)
    attr_cls = dict_to_attr(d)
    return attr_cls

if __name__ == '__main__':
    print(json_to_dict('voiceband_default.json'))
    print(json_to_attr('voiceband_default.json'))
    print(json_to_attr('voiceband_default.json').model_name)
    print(yaml_to_dict('voiceband_default.yaml'))