from models.inception import Inception
from models.nasnet_mobile import NasnetMobile


def create_model(opt):
    if 'inception' in opt.model:
        return Inception(opt)
    elif opt.model == 'nastnetmobile':
        return NasnetMobile(opt)
    else:
        raise Exception("Given model {{{0}}} is not defined.".format(opt.model))
