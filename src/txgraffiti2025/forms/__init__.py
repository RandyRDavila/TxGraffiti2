"""
Unified import layer for all conjecture forms (R₁–R₆).

This makes txgraffiti2025.forms a single access point for:
    - Base logic and expressions        (utils, generic_conjecture, predicates)
    - Algebraic forms                   (linear, nonlinear, floorceil, logexp)
    - Logical and qualitative forms     (class_relations, implication, qualitative, substructure)
"""

from . import utils
from . import predicates
from . import generic_conjecture
from . import linear
from . import nonlinear
from . import floorceil
from . import logexp
from . import implication
from . import qualitative
from . import class_relations
from . import substructure

# Re-export everything explicitly
from .utils import *
from .predicates import *
from .generic_conjecture import *
from .linear import *
from .nonlinear import *
from .floorceil import *
from .logexp import *
from .implication import *
from .qualitative import *
from .class_relations import *
from .substructure import *
