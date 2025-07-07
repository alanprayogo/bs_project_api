# bid_snapper_backend/src/biding_strategies.py

from src.prec.opening import prec_opening
from src.prec.respon_1c import prec_respon_1c
from src.prec.respon_1d import prec_respon_1d
from src.prec.respon_1h import prec_respon_1h
from src.prec.respon_1s import prec_respon_1s
from src.prec.respon_1nt import prec_respon_1nt

BIDING_STRATEGIES = {
    # Precision
    "prec_opening": prec_opening,
    "prec_respon_1c": prec_respon_1c,
    "prec_respon_1d": prec_respon_1d,
    "prec_respon_1h": prec_respon_1h,
    "prec_respon_1s": prec_respon_1s,
    "prec_respon_1nt": prec_respon_1nt,
    # SAYC
    # "sayc_opening": sayc_opening,
    # "sayc_respon_1c": sayc_respon_1c,

    # Tambah skema lain di sini
}