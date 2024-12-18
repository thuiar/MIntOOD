from .MAG_BERT import MAG_BERT
from .MMIM import MMIM
from .MULT import MULT
from .TCL_MAP import TCL_MAP
from .SDIF import SDIF
from .Spectra import Spectra

multimodal_methods_map = {
    'mag_bert': MAG_BERT,
    'mmim': MMIM, 
    'mult': MULT,
    'tcl_map': TCL_MAP,
    'sdif': SDIF,
    'spectra': Spectra,
}