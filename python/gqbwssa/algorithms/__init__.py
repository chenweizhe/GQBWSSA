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
