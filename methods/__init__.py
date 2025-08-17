
from .fedavg import *
from .fedper import *
from .local import *
from .fedmate import *

def local_update(rule):
    LocalUpdate = {'FedAvg':LocalUpdate_FedAvg,
                   'FedPer':LocalUpdate_FedPer,
                   'Local':LocalUpdate_StandAlone,
                   'FedMate':LocalUpdate_FedMate,
    }

    return LocalUpdate[rule]