"""Frozen measured-VNC core with trainable sensory and motor interfaces."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
from vnc_stepper import VNCStepper  # noqa: E402


VNC_CONFIG = {
    "substeps": 4,
    "tau": 0.02,
    "dt": 0.005,
    "gW": 0.03,
    "model": "paper",
    "threshold": 7.5,
    "activation_gain": 1.0,
    "fr_cap": 200.0,
    "heterogeneity": True,
    "seed": 1,
}
COMMAND_MAP_SHA256 = "97e6d46d0605204716accc8fdc6a945b2e16d367d68feaf408f0d0c56fd11cb3"


def _tensor(value, device="cpu") -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device).detach().clone()


# [LG]/[DA] interface capacity. legged_gym and DecAP both run an ELU MLP at
# [512, 256, 128] (legged_robot_config.py:207-209); rsl_rl builds it as
# Linear -> ELU per hidden layer with a bare Linear last (modules/mlp.py:54-73),
# which is what _mlp reproduces.
INTERFACE_HIDDEN_DIMS = (512, 256, 128)


def _mlp(input_dim: int, hidden_dims, output_dim: int) -> nn.Module:
    """rsl_rl's MLP: Linear -> ELU per hidden layer, bare Linear output.

    `hidden_dims` empty gives a single Linear, which is the original interface
    and stays the default -- so `V4_INTERFACE=linear` reproduces every number
    recorded before this existed.
    """
    dims = list(hidden_dims)
    if not dims:
        return nn.Linear(input_dim, output_dim)
    layers: list[nn.Module] = [nn.Linear(input_dim, dims[0]), nn.ELU()]
    for index in range(len(dims) - 1):
        layers += [nn.Linear(dims[index], dims[index + 1]), nn.ELU()]
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


def interface_hidden_dims() -> tuple[int, ...]:
    """Hidden widths for the trainable interface, from V4_INTERFACE.

    "linear" (default) keeps the two bare Linear maps every recorded result was
    produced with. "mlp" adopts [LG]'s [512, 256, 128] on BOTH sides of the core.

    What that changes, and what it deliberately does not:
      * the 4-latent descending bottleneck stays 4. Everything the interface
        tells the connectome still passes through four numbers and the fixed
        anatomical command map.
      * the core stays frozen, zero trainable parameters, hash-checked.
      * the decoder's ONLY input stays the core's motor channels. It has no path
        to the observation, so it cannot route around the connectome -- it can
        only re-read what the connectome produced. `_assert_no_observation_bypass`
        holds this.
    What it does change is honest to state: the interface grows from 4,966
    parameters to roughly 499,000, which is larger than the 485,074-edge frozen
    core it wraps. A large readout on a driven recurrent network is reservoir
    computing, and this project already recorded a moat null for that shape
    (Rung 1). Capacity here buys walking at the cost of how much of the walking
    is attributable to the measured connectome.
    """
    mode = (os.environ.get("V4_INTERFACE") or "linear").strip().lower()
    if mode == "linear":
        return ()
    if mode == "mlp":
        raw = os.environ.get("V4_INTERFACE_HIDDEN")
        if raw:
            return tuple(int(part) for part in raw.replace(" ", "").split(",") if part)
        return INTERFACE_HIDDEN_DIMS
    raise ValueError(f"unknown V4_INTERFACE {mode!r}; expected linear or mlp")


class _FixedSparseMM(torch.autograd.Function):
    """Multiply by a fixed CSR matrix without building gradients for it."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, transpose: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(transpose)
        with torch.no_grad():
            return torch.sparse.mm(weight, value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        (transpose,) = ctx.saved_tensors
        with torch.no_grad():
            value_gradient = torch.sparse.mm(transpose, gradient)
        return None, None, value_gradient


class FrozenVNC(nn.Module):
    """Autograd-safe mirror of ``VNCStepper.step`` with a frozen CSR matrix.

    ``VNCStepper.step`` intentionally runs under ``no_grad`` and owns mutable
    state. This wrapper keeps its exact rate equations while passing state in
    and out, so only the two interface modules receive gradients.
    """

    def __init__(self, data: Path = DATA, frozen_weight: torch.Tensor | None = None, **overrides):
        super().__init__()
        config = dict(VNC_CONFIG)
        config.update(overrides)
        device = torch.device(str(config.get("device", "cpu")))
        if device.type == "mps":
            # No sparse CSR support on MPS (pytorch/pytorch#129842).
            raise ValueError("embody_policy cannot run the sparse core on mps; use cpu or cuda")
        config["device"] = str(device)
        self.device = device
        config["shuffle"] = bool(config.get("shuffle", False))
        self.stepper = VNCStepper(data=Path(data), **config)
        if self.stepper.shuffle:
            raise ValueError("policy scaffold requires the real, unshuffled VNC graph")

        if frozen_weight is None:
            frozen_weight = self.stepper.W
        frozen_weight = frozen_weight.to_sparse_csr().float().to(device).detach().clone()
        if tuple(frozen_weight.shape) != tuple(self.stepper.W.shape):
            raise ValueError(f"frozen_weight has shape {tuple(frozen_weight.shape)}; expected {tuple(self.stepper.W.shape)}")
        self.register_buffer("W", frozen_weight)
        self.register_buffer("W_T", frozen_weight.transpose(0, 1).to_sparse_csr())
        self.register_buffer("descending_in", torch.from_numpy(self.stepper.descending_in.copy()).long().to(device))
        self.register_buffer("motor_out", torch.from_numpy(self.stepper.motor_out.copy()).long().to(device))
        self.register_buffer("decay", _tensor(self.stepper.decay, device))
        self.register_buffer("dt_tau", _tensor(self.stepper.dt_tau, device))
        self.register_buffer("activation_gain", _tensor(self.stepper.activation_gain, device))
        self.register_buffer("threshold", _tensor(self.stepper.threshold, device))
        self.register_buffer("fr_cap", _tensor(self.stepper.fr_cap, device))
        self.n = self.stepper.n
        # One state variable per neuron. The [SH] LIF core in vnc_lif_core carries
        # four, so everything that validates or resets a state tensor reads this
        # rather than assuming it equals n.
        self.state_dim = self.n
        self.substeps = self.stepper.substeps
        self.gW = self.stepper.gW
        self.model = self.stepper.model
        self.W.requires_grad_(False)
        self.eval()

    def initial_state(self, batch_size: int | None = None) -> torch.Tensor:
        shape = (self.n,) if batch_size is None else (self.n, int(batch_size))
        if batch_size is not None and int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        return torch.zeros(shape, dtype=torch.float32, device=self.W.device)

    def forward(self, descending_current: torch.Tensor, node_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.W.device
        current = torch.as_tensor(descending_current, dtype=torch.float32, device=device)
        state = torch.as_tensor(node_state, dtype=torch.float32, device=device)
        if state.ndim == 1:
            if state.shape[0] != self.n or current.ndim != 1:
                raise ValueError(f"single node_state/current shapes must be ({self.n},) and (d,)")
            batched = False
            batch_size = None
        elif state.ndim == 2:
            if state.shape[0] != self.n or state.shape[1] < 1 or current.ndim != 2 or current.shape[1] != state.shape[1]:
                raise ValueError(f"batched node_state/current shapes must be ({self.n}, B) and (d, B)")
            batched = True
            batch_size = state.shape[1]
        else:
            raise ValueError(f"node_state has shape {tuple(state.shape)}; expected ({self.n},) or ({self.n}, B)")

        expected_current = (len(self.descending_in), batch_size) if batched else (len(self.descending_in),)
        full_current = (self.n, batch_size) if batched else (self.n,)
        if tuple(current.shape) == expected_current:
            current_shape = (self.n, batch_size) if batched else (self.n,)
            current = torch.zeros(current_shape, dtype=current.dtype, device=device).index_copy(0, self.descending_in, current)
        elif tuple(current.shape) != full_current:
            expected = f"({self.n}, B) or ({len(self.descending_in)}, B)" if batched else f"{self.n} or {len(self.descending_in)}"
            raise ValueError(f"descending current has shape {tuple(current.shape)}; expected {expected}")

        decay = self.decay.reshape(-1, 1) if batched and self.decay.ndim else self.decay
        dt_tau = self.dt_tau.reshape(-1, 1) if batched and self.dt_tau.ndim else self.dt_tau
        activation_gain = self.activation_gain.reshape(-1, 1) if batched and self.activation_gain.ndim else self.activation_gain
        threshold = self.threshold.reshape(-1, 1) if batched and self.threshold.ndim else self.threshold
        fr_cap = self.fr_cap.reshape(-1, 1) if batched and self.fr_cap.ndim else self.fr_cap

        value = state
        for _ in range(self.substeps):
            value_for_mm = value if batched else value[:, None]
            if self.model == "paper":
                recurrent = _FixedSparseMM.apply(self.W, self.W_T, value_for_mm)
                if not batched:
                    recurrent = recurrent.squeeze(1)
                total = self.gW * recurrent + current
                target = torch.relu(
                    fr_cap * torch.tanh((activation_gain / fr_cap) * (total - threshold))
                )
                value = torch.clamp(value * decay + dt_tau * target, min=0.0)
                value = torch.minimum(value, fr_cap)
            else:
                rates = torch.relu(torch.tanh(value))
                recurrent = _FixedSparseMM.apply(self.W, self.W_T, rates if batched else rates[:, None])
                if not batched:
                    recurrent = recurrent.squeeze(1)
                total = self.gW * recurrent + current
                value = torch.clamp(value * decay + dt_tau * total, -5.0, 5.0)
        if self.model == "paper":
            motor = (value / fr_cap)[self.motor_out]
        else:
            motor = torch.relu(torch.tanh(value))[self.motor_out]
        return value, motor


def vnc_graph_sha256(core: FrozenVNC | VNCStepper) -> str:
    """Hash frozen CSR topology/values and VNC I/O index arrays."""
    W = core.W
    digest = hashlib.sha256()
    digest.update(np.asarray(W.shape, dtype=np.int64).tobytes())
    for value in (W.crow_indices(), W.col_indices(), W.values()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    for name in ("descending_in", "motor_out"):
        array = np.ascontiguousarray(
            getattr(core, name).detach().cpu().numpy() if torch.is_tensor(getattr(core, name)) else getattr(core, name)
        )
        digest.update(name.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class EmbodyPolicy(nn.Module):
    """Trainable body interface around a frozen measured-VNC rate model."""

    def __init__(
        self,
        obs_dim: int,
        data: Path = DATA,
        drive_scale: float = 100.0,
        command_map: np.ndarray | torch.Tensor | None = None,
        leg_posture_map: dict | None = None,
        default_pose: np.ndarray | None = None,
        action_scale: float = 0.5,
        motor_indices: np.ndarray | torch.Tensor | None = None,
        motor_mean: np.ndarray | torch.Tensor | None = None,
        motor_scale: np.ndarray | torch.Tensor | None = None,
        **vnc_overrides,
    ):
        super().__init__()
        if obs_dim < 1 or drive_scale <= 0:
            raise ValueError("obs_dim and drive_scale must be positive")
        self.obs_dim = int(obs_dim)
        self.drive_scale = float(drive_scale)
        # V4_CORE selects the dynamics wrapped around the measured graph. "rate"
        # is what every result so far was produced with; "lif" is Shiu et al.
        # (2024) as published, one free parameter instead of our eight. The graph
        # is byte-identical either way -- only its interpretation changes.
        self.core_kind = (os.environ.get("V4_CORE") or "rate").strip().lower()
        if self.core_kind == "lif":
            from vnc_lif_core import R_POI_MAX, ShiuLIFVNC

            self.core = ShiuLIFVNC(data=data, **vnc_overrides)
            # [SH] drive is a stimulation rate in Hz, capped at the rate their own
            # model uses. Honour an explicit drive_scale; replace the rate core's
            # default, which is millivolt-flavoured and meaningless here.
            if drive_scale == 100.0:
                self.drive_scale = float(R_POI_MAX)
        elif self.core_kind == "rate":
            self.core = FrozenVNC(data=data, **vnc_overrides)
        else:
            raise ValueError(f"unknown V4_CORE {self.core_kind!r}; expected rate or lif")
        if motor_indices is None:
            motor_indices = np.arange(len(self.core.motor_out), dtype=np.int64)
        motor_indices = torch.as_tensor(motor_indices, dtype=torch.long).reshape(-1)
        if motor_indices.numel() < 1 or torch.any(motor_indices < 0) or torch.any(motor_indices >= len(self.core.motor_out)):
            raise ValueError("motor_indices must select at least one valid motor output")
        if torch.unique(motor_indices).numel() != motor_indices.numel():
            raise ValueError("motor_indices must be unique")
        self.register_buffer("motor_indices", motor_indices.clone())
        if (motor_mean is None) != (motor_scale is None):
            raise ValueError("motor_mean and motor_scale must be supplied together")
        if motor_mean is None:
            motor_mean = torch.zeros(motor_indices.numel(), dtype=torch.float32)
            motor_scale = torch.ones(motor_indices.numel(), dtype=torch.float32)
        motor_mean = torch.as_tensor(motor_mean, dtype=torch.float32).reshape(-1)
        motor_scale = torch.as_tensor(motor_scale, dtype=torch.float32).reshape(-1)
        if motor_mean.numel() != motor_indices.numel() or motor_scale.numel() != motor_indices.numel():
            raise ValueError("motor affine parameters must match motor_indices")
        if not torch.isfinite(motor_mean).all() or not torch.isfinite(motor_scale).all() or torch.any(motor_scale <= 0):
            raise ValueError("motor affine parameters must be finite and positive where scaled")
        self.register_buffer("motor_mean", motor_mean.clone())
        self.register_buffer("motor_scale", motor_scale.clone())
        device = self.core.W.device
        if command_map is None:
            # Wide interface: one learned drive per descending neuron. Retained only
            # for the registered Stage-1 comparison; it lets the encoder program the
            # core directly, so any connectivity result becomes an interface result.
            self.register_buffer("command_map", torch.empty(0, device=device))
            self.latent_dim = len(self.core.descending_in)
        else:
            command_map = torch.as_tensor(command_map, dtype=torch.float32)
            if command_map.ndim != 2 or command_map.shape[1] != len(self.core.descending_in):
                raise ValueError(
                    f"command_map has shape {tuple(command_map.shape)}; expected (latent, {len(self.core.descending_in)})"
                )
            if not torch.isfinite(command_map).all():
                raise ValueError("command_map must be finite")
            if torch.linalg.matrix_rank(command_map) != command_map.shape[0]:
                raise ValueError("command_map rows must be linearly independent")
            self.register_buffer("command_map", command_map.clone().to(device))
            self.latent_dim = int(command_map.shape[0])
        self.interface_hidden = interface_hidden_dims()
        # Note the output width: latent_dim, not something wider. Depth is what
        # [LG] has and we did not; the 4-number descending bottleneck is ours and
        # stays, because widening it is what would make the interface the driver.
        self.encoder = _mlp(self.obs_dim, self.interface_hidden, self.latent_dim)
        # Reduced-agency decoder: one scalar per leg, expanded to three joint
        # targets by a STATIC posture map (see build_leg_posture_map.py). The
        # brain still decides when each leg moves; it stops re-deriving how one
        # leg's joints are shaped at a given extension, which is a fact about
        # this body rather than a motor decision. Measured on the teacher, one
        # scalar per leg carries [0.963, 0.987, 0.937, 0.869, 0.929, 0.665] of
        # that leg's joint motion, mean 0.892.
        #
        # The map holds no clock, phase or oscillator: feed it a constant and the
        # legs hold a pose. That is what separates it from the forbidden external
        # CPG, which would supply the rhythm and leave the connectome a speed knob.
        # Default standing pose: the anchor every action is an offset from.
        # legged_gym calls this default_dof_pos; here it is the hexapod's home
        # keyframe, the same stance the teacher holds.
        self.action_scale = float(action_scale)
        default = torch.as_tensor(
            np.asarray(default_pose, dtype=np.float32) if default_pose is not None
            else np.zeros(18, dtype=np.float32))
        if default.shape != (18,):
            raise ValueError(f"default_pose has shape {tuple(default.shape)}; expected (18,)")
        self.register_buffer("default_pose", default.to(device))
        if leg_posture_map is None:
            self.leg_posture = False
            self.decoder = _mlp(motor_indices.numel(), self.interface_hidden, 18)
        else:
            home = torch.as_tensor(leg_posture_map["home"], dtype=torch.float32)
            direction = torch.as_tensor(leg_posture_map["direction"], dtype=torch.float32)
            if home.shape != (18,) or direction.shape != (6, 3):
                raise ValueError(
                    f"leg_posture_map has home {tuple(home.shape)} direction "
                    f"{tuple(direction.shape)}; expected (18,) and (6, 3)")
            if not (torch.isfinite(home).all() and torch.isfinite(direction).all()):
                raise ValueError("leg_posture_map must be finite")
            self.leg_posture = True
            self.register_buffer("posture_home", home.to(device))
            self.register_buffer("posture_direction", direction.to(device))
            self.decoder = _mlp(motor_indices.numel(), self.interface_hidden, 6)
        self.to(device)
        assert not isinstance(self.core.W, nn.Parameter)
        assert not self.core.W.requires_grad
        assert not tuple(self.core.parameters())
        assert not isinstance(self.command_map, nn.Parameter)
        assert not self.command_map.requires_grad

    def initial_state(self) -> torch.Tensor:
        return self.core.initial_state()

    def initial_state_batch(self, batch_size: int) -> torch.Tensor:
        return self.core.initial_state(batch_size)

    def reset_state_batch(self, node_state: torch.Tensor, done: np.ndarray | torch.Tensor) -> torch.Tensor:
        state = torch.as_tensor(node_state, dtype=torch.float32, device=self.core.W.device)
        mask = torch.as_tensor(done, dtype=torch.bool, device=state.device).reshape(-1)
        if state.ndim != 2 or state.shape != (self.core.state_dim, mask.numel()):
            raise ValueError(f"batched node_state has shape {tuple(state.shape)}; expected ({self.core.state_dim}, B)")
        if state.requires_grad:
            state = state.clone()
        if mask.any():
            state[:, mask] = 0.0
            assert float(state[:, mask].abs().max().detach().cpu()) == 0.0
        return state

    def _to_joints(self, decoded: torch.Tensor) -> torch.Tensor:
        """Decoder output -> 18 absolute joint targets.

        The output is an OFFSET FROM THE DEFAULT STANDING POSE, not an absolute
        target. This is legged_gym's contract, legged_robot.py:368:

            torques = p_gains*(actions_scaled + default_dof_pos - dof_pos) - d_gains*dof_vel
            actions_scaled = actions * action_scale        # action_scale = 0.5

        It matters more than it looks. With absolute targets a freshly initialised
        decoder commands an arbitrary pose and the robot collapses at once: measured
        on the 2.0M-step checkpoint, episodes lasted 0.33 s with the mean action and
        0.31 s with sampled actions -- about 16 control steps. PPO was learning from
        16-step episodes and could never experience walking, which is why more budget
        never helped. Anchored at the default pose, a random policy STANDS, and
        training starts from a robot that is upright.
        """
        if self.leg_posture:
            legs = decoded.unsqueeze(-1) * self.posture_direction
            deviation = legs.reshape(*decoded.shape[:-1], 18)
            anchor = self.posture_home
        else:
            deviation = decoded
            anchor = self.default_pose
        return anchor + self.action_scale * deviation

    def interface_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.encoder.parameters()) + tuple(self.decoder.parameters())

    def step(self, obs: torch.Tensor | np.ndarray, node_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observation = torch.as_tensor(obs, dtype=torch.float32, device=self.core.W.device)
        if observation.ndim == 2 and observation.shape[0] == 1:
            observation = observation[0]
        elif observation.ndim == 2:
            return self.step_batch(observation, node_state)
        if observation.ndim != 1 or observation.numel() != self.obs_dim:
            raise ValueError(f"obs has shape {tuple(observation.shape)}; expected ({self.obs_dim},)")
        latent = self.encoder(observation)
        command = latent if self.command_map.numel() == 0 else latent @ self.command_map
        drive = self.drive_scale * torch.sigmoid(command)
        new_state, motor = self.core(drive, node_state)
        motor = motor[self.motor_indices]
        motor = self.motor_mean + (motor - self.motor_mean) * self.motor_scale
        return self._to_joints(self.decoder(motor)), new_state

    def step_batch(self, obs: torch.Tensor | np.ndarray, node_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observations = torch.as_tensor(obs, dtype=torch.float32, device=self.core.W.device)
        state = torch.as_tensor(node_state, dtype=torch.float32, device=self.core.W.device)
        if observations.ndim != 2 or observations.shape[1] != self.obs_dim or observations.shape[0] < 1:
            raise ValueError(f"batched obs has shape {tuple(observations.shape)}; expected (B, {self.obs_dim})")
        if state.ndim != 2 or state.shape != (self.core.state_dim, observations.shape[0]):
            raise ValueError(f"batched node_state has shape {tuple(state.shape)}; expected ({self.core.state_dim}, B)")
        latent = self.encoder(observations)
        command = latent if self.command_map.numel() == 0 else latent @ self.command_map
        drive = self.drive_scale * torch.sigmoid(command)
        new_state, motor = self.core(drive.transpose(0, 1), state)
        motor = motor[self.motor_indices, :]
        motor = self.motor_mean[:, None] + (motor - self.motor_mean[:, None]) * self.motor_scale[:, None]
        return self._to_joints(self.decoder(motor.transpose(0, 1))), new_state


def load_command_map(path: Path | None = None) -> np.ndarray:
    """Load the fixed anatomical descending-command map built by build_dn_command_map.py."""
    path = path or (HERE.parent.parent / "results" / "path2" / "dn_command_map.npz")
    with np.load(path, allow_pickle=True) as bundle:
        matrix = bundle["matrix"].astype(np.float32)
    digest = command_map_sha256(matrix)
    if matrix.shape != (4, 1325) or digest != COMMAND_MAP_SHA256:
        raise ValueError(f"command map changed: shape={matrix.shape} sha256={digest}")
    return matrix


def command_map_sha256(value: np.ndarray | torch.Tensor) -> str:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value, dtype=np.float32)
    array = np.asarray(array, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _output_width(module: nn.Module) -> int:
    """Output width of a Linear or of an MLP built by `_mlp`."""
    if isinstance(module, nn.Linear):
        return module.out_features
    last = [m for m in module.modules() if isinstance(m, nn.Linear)][-1]
    return last.out_features


def _assert_no_observation_bypass(policy: EmbodyPolicy) -> None:
    """The decoder must read the frozen core and nothing else.

    This is the invariant that survives giving the interface [LG]'s capacity. A
    decoder with 300k parameters is only a body interface for as long as its
    input is the connectome's motor channels; the moment it can also see the
    observation it can route around the connectome entirely and the result stops
    being about the connectome at all.
    """
    import inspect

    first = policy.decoder if isinstance(policy.decoder, nn.Linear) else policy.decoder[0]
    expected = int(policy.motor_indices.numel())
    assert isinstance(first, nn.Linear) and first.in_features == expected, (
        f"decoder takes {getattr(first, 'in_features', '?')} inputs; the core "
        f"supplies {expected}"
    )
    for name in ("step", "step_batch"):
        source = inspect.getsource(getattr(EmbodyPolicy, name))
        assert "self.decoder(motor" in source, f"{name} must decode the core's motor output"
        assert "self.decoder(obs" not in source, f"{name} feeds the observation to the decoder"


def _assert_frozen(policy: EmbodyPolicy, optimizer: torch.optim.Optimizer) -> None:
    assert not policy.core.W.requires_grad
    assert not isinstance(policy.core.W, nn.Parameter)
    assert not policy.command_map.requires_grad
    assert not isinstance(policy.command_map, nn.Parameter)
    core_parameter_ids = {id(parameter) for parameter in policy.core.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert not core_parameter_ids & optimizer_ids
    assert {id(parameter) for parameter in policy.interface_parameters()} == optimizer_ids


def resolve_device(requested: str = "auto") -> str:
    """cpu or cuda only: the sparse core cannot run on mps (pytorch/pytorch#129842)."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "mps":
        raise ValueError("embody_policy cannot run the sparse core on mps; use cpu or cuda")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device {requested!r}; use cpu, cuda, or auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return str(device)


def self_check(device: str = "auto") -> dict[str, object]:
    from embody_env import HexapodEnv

    device = resolve_device(device)
    env = HexapodEnv(max_steps=2)
    obs = env.reset()
    command_map = load_command_map()
    policy = EmbodyPolicy(obs_dim=obs.shape[0], command_map=command_map, device=device)
    assert policy.latent_dim == command_map.shape[0]
    assert _output_width(policy.encoder) == command_map.shape[0]
    optimizer = torch.optim.Adam(policy.interface_parameters(), lr=1e-3)
    _assert_frozen(policy, optimizer)
    _assert_no_observation_bypass(policy)
    before = vnc_graph_sha256(policy.core)
    action, node_state = policy.step(obs, policy.initial_state())
    assert tuple(action.shape) == (18,)
    assert tuple(node_state.shape) == (policy.core.state_dim,)
    assert torch.isfinite(action).all() and torch.isfinite(node_state).all()
    next_obs, reward, done = env.step(action.detach().cpu().numpy())
    assert next_obs.shape == obs.shape and np.isfinite(next_obs).all() and np.isfinite(reward)
    assert isinstance(done, bool)

    batch_size = 3
    generator = torch.Generator(device=policy.core.W.device).manual_seed(20260831)
    base_current = torch.rand((len(policy.core.descending_in), batch_size), generator=generator, device=device)
    base_state = torch.rand((policy.core.state_dim, batch_size), generator=generator, device=device)
    batch_current = base_current.detach().clone().requires_grad_()
    batch_state = base_state.detach().clone().requires_grad_()
    batch_value, batch_motor = policy.core(batch_current, batch_state)
    single_values, single_motors = [], []
    single_currents, single_states = [], []
    for index in range(batch_size):
        current = base_current[:, index].detach().clone().requires_grad_()
        state = base_state[:, index].detach().clone().requires_grad_()
        value, motor = policy.core(current, state)
        single_values.append(value)
        single_motors.append(motor)
        single_currents.append(current)
        single_states.append(state)
    single_value = torch.stack(single_values, dim=1)
    single_motor = torch.stack(single_motors, dim=1)
    equivalence_max_abs_diff = max(
        float((batch_value - single_value).detach().abs().max()),
        float((batch_motor - single_motor).detach().abs().max()),
    )
    equivalence_tolerance = 1e-5
    assert equivalence_max_abs_diff <= equivalence_tolerance, (
        f"batched core differs from single path by {equivalence_max_abs_diff:.3g} > {equivalence_tolerance}"
    )
    batch_loss = batch_value.square().sum() + batch_motor.square().sum()
    single_loss = single_value.square().sum() + single_motor.square().sum()
    batch_loss.backward()
    single_loss.backward()
    single_current_gradient = torch.stack([current.grad for current in single_currents], dim=1)
    single_state_gradient = torch.stack([state.grad for state in single_states], dim=1)
    gradient_max_abs_diff = max(
        float((batch_current.grad - single_current_gradient).detach().abs().max()),
        float((batch_state.grad - single_state_gradient).detach().abs().max()),
    )
    assert torch.isfinite(batch_current.grad).all() and torch.isfinite(batch_state.grad).all()
    assert gradient_max_abs_diff <= equivalence_tolerance, (
        f"batched backward differs from single path by {gradient_max_abs_diff:.3g} > {equivalence_tolerance}"
    )
    assert before == vnc_graph_sha256(policy.core)
    assert policy.core.W.grad is None
    interface_parameters = sum(p.numel() for p in policy.interface_parameters())
    return {
        "device": device,
        "latent_dim": policy.latent_dim,
        "interface_parameters": interface_parameters,
        "interface_hidden": list(policy.interface_hidden),
        "core_kind": policy.core_kind,
        "obs_shape": tuple(obs.shape),
        "action_shape": tuple(action.shape),
        "node_state_shape": tuple(node_state.shape),
        "core_sha256": before,
        "done_after_one_step": done,
        "batched_core_shape": tuple(batch_value.shape),
        "batched_motor_shape": tuple(batch_motor.shape),
        "batched_equivalence_max_abs_diff": equivalence_max_abs_diff,
        "batched_backward_max_abs_diff": gradient_max_abs_diff,
        "batched_equivalence_tolerance": equivalence_tolerance,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    result = self_check(parser.parse_args().device)
    print(
        "FROZEN-CORE POLICY: PASS "
        f"(device={result['device']} latent_dim={result['latent_dim']} interface_params={result['interface_parameters']} "
        f"obs_shape={result['obs_shape']} action_shape={result['action_shape']} "
        f"node_state_shape={result['node_state_shape']} core_sha256={result['core_sha256']} "
        f"done_after_one_step={result['done_after_one_step']} "
        f"batched_max_abs_diff={result['batched_equivalence_max_abs_diff']:.3g} "
        f"backward_max_abs_diff={result['batched_backward_max_abs_diff']:.3g})"
    )
