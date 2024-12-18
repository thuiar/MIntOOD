from .MAG_BERT.manager import MAG_BERT
from .TEXT.manager import TEXT
from .MMIM.manager import MMIM
from .MIntOOD.manager import MIntOOD
from .MULT.manager import MULT
from .TCL_MAP.manager import TCL_MAP
from .SDIF.manager import SDIF
from .Spectra.manager import Spectra



method_map = {
    'mag_bert': MAG_BERT,
    'text': TEXT,
    'mmim': MMIM,
    'mintood': MIntOOD,
    'mult': MULT,
    'tcl_map': TCL_MAP,
    'sdif': SDIF,
    'spectra': Spectra,
}