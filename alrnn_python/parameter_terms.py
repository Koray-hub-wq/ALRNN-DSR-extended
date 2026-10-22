"""Registry of optional phi-dependent contributions to the AL-RNN recurrence."""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TermWeight:
    name: str
    axes: tuple[str, str] = ("latent", "phi")
    init_std: float = 0.0

    def shape(self, latent_dim, phi_dim):
        sizes = {"latent": latent_dim, "phi": phi_dim}
        return tuple(sizes[axis] for axis in self.axes)


@dataclass(frozen=True)
class ParameterTerm:
    parameter_name: str
    contribution: Callable
    use_activated_state: bool = False
    extra_weights: tuple[TermWeight, ...] = ()
    needs_activation: bool = False

    @property
    def weights(self):
        return (TermWeight(self.parameter_name), *self.extra_weights)


def additive_input(z, phi, weights):
    return phi @ weights.t()


def diagonal_coupling(z, phi, weights):
    return z * (phi @ weights.t())


def shifted_relu(z, phi, shift, projection, *, activate):
    return activate(z + phi @ shift.t()) @ projection.t()


# Every term has a primary (M, phi_dim) weight and optional declared extra weights.
# Keep this module independent of torch so the CLI can set thread limits first.
PARAMETER_TERMS = {
    "additive": ParameterTerm("C", additive_input),
    "diagonal": ParameterTerm("D_phi", diagonal_coupling),
    "diagonal-relu": ParameterTerm("D_relu_phi", diagonal_coupling, use_activated_state=True),
    "shifted-relu": ParameterTerm(
        "Q_phi", shifted_relu,
        extra_weights=(TermWeight("V_phi", axes=("latent", "latent"), init_std=0.01),),
        needs_activation=True,
    ),
}


def resolve_phi_terms(requested=None, use_phi=False):
    if requested is None:
        return ("additive",) if use_phi else ()
    requested = tuple(requested)
    if "none" in requested:
        if requested != ("none",):
            raise ValueError("The phi term 'none' cannot be combined with other terms.")
        return ()
    if len(requested) != len(set(requested)):
        raise ValueError("Each phi term may only be selected once.")
    unknown = set(requested) - PARAMETER_TERMS.keys()
    if unknown:
        raise ValueError(f"Unknown phi terms: {sorted(unknown)}")
    return tuple(name for name in PARAMETER_TERMS if name in requested)
