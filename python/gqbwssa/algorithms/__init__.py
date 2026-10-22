from .gqbwssa import gqbwssa
from .dbo import dbo
from .gwo import gwo
from .hho import hho
from .woa import woa

ALGORITHMS = {
    "GWO": gwo,
    "WOA": woa,
    "DBO": dbo,
    "HHO": hho,
    "GQBWSSA": gqbwssa,
}


DISPLAY_NAMES = {
    "GWO": "GWO-RF",
    "WOA": "WOA-RF",
    "DBO": "DBO-RF",
    "HHO": "HHO-RF",
    "GQBWSSA": "GQBWSSA-RF",
}
