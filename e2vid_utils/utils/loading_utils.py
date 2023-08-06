import torch
from e2vid_utils.model.model import *

def load_E2VID(path_to_model, device=torch.device('cpu')): 
    print('Loading model {}...'.format(path_to_model))
    raw_model = torch.load(path_to_model, map_location=device)
    arch = raw_model['arch']
    try:
        model_type = raw_model['model']
    except KeyError:
        model_type = raw_model['config']['model']
    # instantiate model
    model = eval(arch)(model_type)
    # load model weights    
    model.load_state_dict(raw_model['state_dict'])
    return model

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    return device
