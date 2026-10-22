"""Shared AL-RNN dynamics for training and checkpoint-based evaluation."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from parameter_terms import PARAMETER_TERMS, resolve_phi_terms


class ALRNNRecurrence(nn.Module):
    def activate(self, z):
        activated = z.clone()
        if self.P:
            activated[:, -self.P:] = torch.relu(activated[:, -self.P:])
        return activated

    def configure(self, M, P, N, phi_dim, phi_terms):
        if not 0 < N <= M or not 0 <= P <= M or phi_dim < 0:
            raise ValueError("Require 0 < N <= M, 0 <= P <= M and phi_dim >= 0.")
        self.M, self.P, self.N = int(M), int(P), int(N)
        self.phi_terms = resolve_phi_terms(phi_terms)
        if self.phi_terms and phi_dim == 0:
            raise ValueError("Active phi terms require phi_dim > 0.")
        self.phi_dim = int(phi_dim) if self.phi_terms else 0

    def forward(self, z, phi_t=None):
        if z.ndim != 2 or z.shape[1] != self.M:
            raise ValueError(f"Expected z with shape (batch, {self.M}), got {tuple(z.shape)}")
        if self.phi_terms:
            if phi_t is None or phi_t.ndim != 2 or phi_t.shape != (z.shape[0], self.phi_dim):
                shape = None if phi_t is None else tuple(phi_t.shape)
                raise ValueError(f"Expected phi_t with shape ({z.shape[0]}, {self.phi_dim}), got {shape}")
        activated = self.activate(z)
        out = self.A * z + activated @ self.W.t() + self.h
        for name in self.phi_terms:
            term = PARAMETER_TERMS[name]
            state = activated if term.use_activated_state else z
            weights = [getattr(self, weight.name) for weight in term.weights]
            kwargs = {"activate": self.activate} if term.needs_activation else {}
            out = out + term.contribution(state, phi_t, *weights, **kwargs)
        return out

    def model_config(self):
        return dict(schema_version=1, model="AL_RNN", latent_dim=self.M, pwl_units=self.P,
                    observed_dim=self.N, phi_dim=self.phi_dim, phi_terms=list(self.phi_terms))


class AL_RNN(ALRNNRecurrence):
    def __init__(self, M, P, N, phi_dim=0, phi_terms=None):
        super().__init__()
        terms = resolve_phi_terms(phi_terms, use_phi=phi_dim > 0)
        self.configure(M, P, N, phi_dim, terms)
        self.A, self.W, self.h = self.initialize_AWh_random()
        self.B = self.init_uniform((self.N, self.M))
        for name, term in PARAMETER_TERMS.items():
            for weight in term.weights:
                value = None
                if name in self.phi_terms:
                    shape = weight.shape(self.M, self.phi_dim)
                    tensor = torch.randn(*shape) * weight.init_std if weight.init_std else torch.zeros(shape)
                    value = nn.Parameter(tensor)
                self.register_parameter(weight.name, value)

    def initialize_AWh_random(self):
        A = nn.Parameter(torch.diagonal(self.normalized_positive_definite(self.M), 0))
        W = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        h = nn.Parameter(torch.zeros(self.M))
        return A, W, h

    def normalized_positive_definite(self, M):
        R = np.random.randn(M, M).astype(np.float32)
        K = np.matmul(R.T, R) / M + np.eye(M)
        lambda_max = np.max(np.abs(np.linalg.eigvals(K)))
        return torch.tensor(K / lambda_max).float()

    def init_uniform(self, shape):
        tensor = torch.empty(*shape)
        radius = 1 / math.sqrt(shape[0])
        torch.nn.init.uniform_(tensor, -radius, radius)
        return nn.Parameter(tensor, requires_grad=True)


class SavedALRNN(ALRNNRecurrence):
    """Load tensors directly without random initialization or optimizer parameters."""

    def __init__(self, state, pwl_units):
        super().__init__()
        required = {"A", "W", "h", "B"}
        term_keys = {weight.name for term in PARAMETER_TERMS.values() for weight in term.weights}
        if not required <= set(state) or set(state) - required - term_keys:
            raise ValueError(f"Unexpected single AL-RNN state_dict keys: {sorted(state)}")
        terms = []
        for name, term in PARAMETER_TERMS.items():
            keys = {weight.name for weight in term.weights}
            if keys & state.keys():
                if not keys <= state.keys():
                    raise ValueError(f"Incomplete {name} checkpoint: missing {sorted(keys - state.keys())}")
                terms.append(name)
        terms = tuple(terms)
        phi_dims = set()
        for name in terms:
            tensor = state[PARAMETER_TERMS[name].parameter_name]
            if tensor.ndim != 2 or tensor.shape[1] == 0:
                raise ValueError(f"Invalid phi weight shape for {name}: {tuple(tensor.shape)}")
            phi_dims.add(tensor.shape[1])
        if len(phi_dims) > 1:
            raise ValueError("All phi terms must use the same phi_dim.")
        if state["A"].ndim != 1 or state["B"].ndim != 2:
            raise ValueError("Invalid A or B shape in checkpoint.")
        self.configure(state["A"].numel(), int(pwl_units), state["B"].shape[0], next(iter(phi_dims), 0), terms)
        shapes = {"A": (self.M,), "W": (self.M, self.M), "h": (self.M,), "B": (self.N, self.M)}
        for name in terms:
            for weight in PARAMETER_TERMS[name].weights:
                shapes[weight.name] = weight.shape(self.M, self.phi_dim)
        for key, shape in shapes.items():
            if tuple(state[key].shape) != shape or not torch.isfinite(state[key]).all():
                raise ValueError(f"Invalid or non-finite saved tensor {key}.")
            self.register_buffer(key, state[key])
        for name, term in PARAMETER_TERMS.items():
            if name not in terms:
                for weight in term.weights:
                    self.register_buffer(weight.name, None)


def load_saved_model(run_dir, device="cpu"):
    run_dir = Path(run_dir)
    with (run_dir / "metrics.json").open(encoding="utf-8-sig") as stream:
        metrics = json.load(stream)
    saved_args = metrics.get("args", {})
    config = metrics.get("model_config", {})
    pwl_units = config.get("pwl_units", saved_args.get("pwl_units", metrics.get("pwl_units")))
    if pwl_units is None:
        raise ValueError(f"Cannot infer pwl_units from {run_dir / 'metrics.json'}")
    model_path = run_dir / "model.pt"
    state = torch.load(model_path, map_location=device, weights_only=True)
    model = SavedALRNN(state, pwl_units).eval()
    expected = model.model_config()
    for key, stored in config.items():
        if key not in expected or stored != expected[key]:
            raise ValueError(f"Checkpoint disagrees with saved model_config.{key}: {stored!r}")
    if saved_args.get("latent_dim", model.M) != model.M or metrics.get("phi_dim", model.phi_dim) != model.phi_dim:
        raise ValueError(f"Checkpoint dimensions disagree with metadata in {run_dir}")
    if saved_args.get("phi_terms") is not None and resolve_phi_terms(saved_args["phi_terms"]) != model.phi_terms:
        raise ValueError(f"Checkpoint phi terms disagree with metadata in {run_dir}")
    return model, metrics, model_path
