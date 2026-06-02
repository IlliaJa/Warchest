"""Vanilla unit classes, generated one-per-type from the roster.

Every Phase-3 unit behaves identically (move/attack/control/deploy/bolster); the
only per-type differences are id and icon, which come straight from
`roster.UNIT_TYPES`. Generating the classes keeps that data in one place instead
of a file per unit. Tactics/attributes (Phase 4) will give some of these classes
real method overrides.
"""
from .baseunit import BaseUnit
from ..roster import UNIT_TYPES

# coin id -> unit class
UNIT_CLASS_BY_ID = {}
for _ut in UNIT_TYPES:
    _cls = type(_ut.name.replace(' ', ''), (BaseUnit,), {'id': _ut.id, 'icon': _ut.icon})
    UNIT_CLASS_BY_ID[_ut.id] = _cls
    globals()[_cls.__name__] = _cls  # export by class name (Swordsman, Knight, ...)

__all__ = ['BaseUnit', 'UNIT_CLASS_BY_ID'] + [c.__name__ for c in UNIT_CLASS_BY_ID.values()]
