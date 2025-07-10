# bid_snapper_backend/src/biding_strategies.py

from src.prec.opening import prec_opening
from src.prec.respon_1c import prec_respon_1c
from src.prec.respon_1d import prec_respon_1d
from src.prec.respon_1h import prec_respon_1h
from src.prec.respon_1s import prec_respon_1s
from src.prec.respon_1nt import prec_respon_1nt
from src.prec.respon_2c import prec_respon_2c
from src.prec.respon_2d import prec_respon_2d
from src.prec.respon_2h import prec_respon_2h
from src.prec.respon_2s import prec_respon_2s

BIDING_STRATEGIES = {
    # Precision
    "prec_opening": prec_opening,
    "prec_respon_1c": prec_respon_1c,
    "prec_respon_1d": prec_respon_1d,
    "prec_respon_1h": prec_respon_1h,
    "prec_respon_1s": prec_respon_1s,
    "prec_respon_1nt": prec_respon_1nt,
    "prec_respon_2c": prec_respon_2c,
    "prec_respon_2d": prec_respon_2d,
    "prec_respon_2h": prec_respon_2h,
    "prec_respon_2s": prec_respon_2s,
    
    # SAYC
    # "sayc_opening": sayc_opening,
    # "sayc_respon_1c": sayc_respon_1c,

    # Tambah skema lain di sini
}