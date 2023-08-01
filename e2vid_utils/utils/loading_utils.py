import torch
from e2vid_utils.model.model import *


def load_E2VID(path_to_model): #, path, train):
    print('Loading model {}...'.format(path_to_model))
    raw_model = torch.load(path_to_model)
    arch = raw_model['arch']

    try:
        model_type = raw_model['model']
    except KeyError:
        model_type = raw_model['config']['model']

    # instantiate model
    model = eval(arch)(model_type)

    # load model weights
    #if train==0:
    
    model.load_state_dict(raw_model['state_dict'])
    
    """
    else:
       pretrained_dict = torch.load(path)
       model_dict = model.state_dict()
       pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
       model_dict.update(pretrained_dict)
       model.load_state_dict(model_dict)
    """
    return model


def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    return device
