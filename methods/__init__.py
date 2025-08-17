
from .fedavg import *
from .fedper import *
from .local import *
from .fedcode_0325 import *

def local_update(rule):
    LocalUpdate = {'FedAvg':LocalUpdate_FedAvg,
                   'FedPer':LocalUpdate_FedPer,
                   'Local':LocalUpdate_StandAlone,
                   'FedCode3':LocalUpdate_FedCode_0325,
    }

    return LocalUpdate[rule]