"""GeoDisaster-FM Dispatcher — Layer 2 (neuro-symbolic reasoner) + Layer 3 (RL policy).

This subpackage implements the AI emergency dispatcher proposed in
``NATURE_PITCH.md``. It sits on top of the perception layer
(geodisaster.models / geodisaster.train) and converts pixel-level disaster
predictions into structured answers to standard emergency-response
questions.

Layer 2 (this file's main scope): neuro-symbolic reasoner.
Layer 3 (geodisaster.dispatch.rl): the RL policy that decides actions
(label, alert, dispatch) — added as the disaster atlas is curated.
"""
from .reasoner import EmergencyReasoner, ReasonerReport
from .queries import STANDARD_QUERIES, EmergencyQuery

__all__ = [
    "EmergencyReasoner",
    "ReasonerReport",
    "STANDARD_QUERIES",
    "EmergencyQuery",
]
