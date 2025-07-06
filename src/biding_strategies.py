# bid_snapper_backend/src/biding_strategies.py

from src.prec.opening import prec_opening
from src.prec.respon_1c import prec_respon_1c

BIDING_STRATEGIES = {
    # Precision Club
    "prec_opening": prec_opening,
    "prec_respon_1c": prec_respon_1c,

    # SAYC
    # "sayc_opening": sayc_opening,
    # "sayc_respon_1c": sayc_respon_1c,

    # Tambah skema lain di sini
}