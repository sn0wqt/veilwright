"""Veilwright — semantic anonymization toolkit and adversarial loop."""

from veilwright.defender import Defender, DefenderError, run_anonymizer
from veilwright.attacker import Attacker, AttackerError
from veilwright.utility import UtilityJudge, UtilityError
from veilwright.orchestrator import run_adversarial_loop
from veilwright.types import (
    DefenderInput,
    DefenderOutput,
    StrategyRecord,
    UtilityInput,
    UtilityOutput,
    AttackerOutput,
    AdversarialIteration,
    AdversarialResult,
)


__all__ = [
    "Defender",
    "DefenderError",
    "Attacker",
    "AttackerError",
    "AttackerOutput",
    "UtilityJudge",
    "UtilityError",
    "DefenderInput",
    "DefenderOutput",
    "StrategyRecord",
    "UtilityInput",
    "UtilityOutput",
    "AdversarialIteration",
    "AdversarialResult",
    "run_anonymizer",
    "run_adversarial_loop",
]
