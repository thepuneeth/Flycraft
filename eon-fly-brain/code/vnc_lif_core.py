"""Shiu et al. (2024) leaky integrate-and-fire dynamics on the frozen VNC graph.

[SH] Shiu, Sterne, Spiller et al., "A Drosophila computational brain model
     reveals sensorimotor processing", Nature 634:210-219 (2024).
     Reference code: github.com/philshiu/Drosophila_brain_model (model.py).

Why this exists
---------------
`embody_policy.FrozenVNC` runs a rate model whose constants we chose: substeps,
gW, threshold 7.5, fr_cap 200, a 100 * sigmoid drive scale. The connectivity is
measured but the dynamics are ours, so every one of those numbers is a number we
have to defend, and a reviewer is entitled to ask whether the result is a fact
about the connectome or about the eight knobs wrapped around it.

[SH] has ONE free parameter, `w_syn`. Every other constant is taken from
published electrophysiology, cited inline below. The synaptic weight is
`w_syn * signed synapse count` -- and `fetch_manc.py:310` already stores exactly
that product of raw synapse counts and predicted-neurotransmitter sign, so this
core reads the SAME matrix `FrozenVNC` does and reinterprets its values in mV
rather than rebuilding anything.

Differences from the reference, and why
---------------------------------------
* brian2 integrates one network at a time; PPO needs 512 in parallel on one
  device, so this is the same equations written as batched tensor ops. The
  integrator is exponential-Euler, which is exact for the linear subsystem over
  a step -- what brian2's `method='linear'` does -- so `dt` sets spike-time
  resolution, not stability, and 2 ms costs nothing in accuracy we can observe
  through a 20 ms control step.
* the 1.8 ms synaptic delay is quantised to one substep (2 ms). Resolving it
  exactly needs dt=0.2 ms and ten times the compute for a 0.2 ms difference the
  control step cannot see.
* [SH] drive a neuron with a Poisson process of weight `w_syn * f_poi` = 68.75 mV
  and set that neuron's refractory period to zero (`model.py:85-92`). One event
  is ten times the 7 mV threshold and nothing holds the neuron out afterwards, so
  a stimulated neuron emits one spike per event: its rate IS the requested rate.
  Here the process is thinned onto the substep grid -- one Bernoulli draw per
  substep at p = rate * dt, exact to O(p^2) -- with a straight-through gradient
  to p, so the encoder's output stays a stimulation rate in Hz, bounded by the
  150 Hz [SH] use, and stays differentiable, which a Poisson sample is not.
  Substituting the process by its MEAN CURRENT instead, which this core did until
  the reference implementation was read, does not preserve any of that: the mean
  sits permanently over threshold, the neuron fires every substep regardless of
  the request, and 50 / 100 / 150 Hz all came out as 341 / 489 / 498 Hz.
* spikes are non-differentiable, and the encoder sits UPSTREAM of the core, so
  PPO's gradient has to cross every substep. The forward pass is a true
  Heaviside; the backward pass uses the standard fast-sigmoid surrogate. The
  core's values are untouched by this -- it is frozen, and holds no parameters.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

# --- [SH] default_params, verbatim (mV, ms, Hz) ------------------------------
# Kakaria and de Bivort 2017  https://doi.org/10.3389/fnbeh.2017.00008
V_0 = -52.0        # resting potential
V_RST = -52.0      # reset potential after a spike
V_TH = -45.0       # spike threshold
T_MBR = 20.0       # membrane time scale
# Juergensen et al  https://doi.org/10.1088/2634-4386/ac3ba6
TAU = 5.0          # synaptic time constant
# Lazar et al  https://doi.org/10.7554/eLife.62362
T_RFC = 2.2        # refractory period
# Paul et al 2015  doi: 10.3389/fncel.2015.00029
T_DLY = 1.8        # synaptic delay
# The single free parameter.
W_SYN = 0.275      # mV per synapse
# [SH]'s optogenetic-activation convention.
F_POI = 250.0      # scaling factor for the Poisson synapse
R_POI_MAX = 150.0  # activation rate (Hz)

# Everything below works in u = v - V_0. V_RST equals V_0, so rest AND reset are
# both u = 0, which lets the packed state tensor reset to zeros exactly the way
# the rate core's does -- `EmbodyPolicy.reset_state_batch` needs no special case.
U_TH = V_TH - V_0                 # 7.0 mV above rest
# [SH]'s refractory period bounds a neuron at 1000/2.2 = 454.5 Hz, but a discrete
# integrator can only hold a neuron out for a whole number of substeps, so the
# ceiling this core actually enforces is 1000 / (ceil(T_RFC/dt) * dt) -- 250 Hz at
# dt = 2 ms. Normalising the motor readout by the ideal 454.5 while the dynamics
# top out at 250 shrank every motor value by 1.8x for no reason present in the
# model, so the divisor is derived per instance as `rate_ceiling_hz`; this stays
# as the continuous-time bound the quantised one is measured against.
RATE_CEILING_HZ = 1000.0 / T_RFC  # 454.5 Hz; the continuous-time bound

STATE_BLOCKS = 4                  # u, g, spikes, refractory countdown


class ShiuLIFVNC(nn.Module):
    """Frozen connectome under [SH] dynamics, interface-compatible with FrozenVNC.

    State is packed as one (STATE_BLOCKS * n, B) tensor so that the existing
    rollout, checkpointing and reset plumbing carries it unchanged.
    """

    def __init__(self, data: Path, weights_name: str = "vnc_weight_csr.pkl",
                 index_name: str = "vnc_io_index.npz", device: str = "cpu",
                 frozen_weight: torch.Tensor | None = None,
                 dt_ms: float = 2.0, control_dt_ms: float = 20.0,
                 w_syn_mv: float = W_SYN, surrogate_slope: float = 0.01,
                 detach_recurrent: bool = False, detach_reset: bool = True,
                 shuffle: bool = False, seed: int = 1, **_ignored) -> None:
        super().__init__()
        import pickle

        if shuffle:
            raise ValueError("policy scaffold requires the real, unshuffled VNC graph")
        data = Path(data)
        torch_device = torch.device(device)
        if torch_device.type == "mps":
            raise ValueError("the sparse core cannot run on mps; use cpu or cuda")
        device = torch_device

        # `frozen_weight` is how the rewired and random control arms hand in their
        # own graph (audit_embody_3way.condition_stepper). Either way the values
        # arriving here are signed synapse counts.
        if frozen_weight is None:
            with (data / weights_name).open("rb") as handle:
                counts = frozen_weight = pickle.load(handle)
        counts = frozen_weight
        counts = counts.to_sparse_csr().float() if counts.layout != torch.sparse_csr else counts.float()
        self.n = int(counts.shape[0])
        self.state_dim = STATE_BLOCKS * self.n

        io = np.load(data / index_name)
        if int(io["n"]) != self.n:
            raise ValueError(f"VNC matrix/index mismatch: {self.n} != {int(io['n'])}")

        # The ONLY transformation of the measured graph: synapse counts become
        # millivolts. Signs already carry the predicted neurotransmitter
        # (fetch_manc.py:310), so nothing about the topology or polarity moves.
        weights = torch.sparse_csr_tensor(
            counts.crow_indices(), counts.col_indices(),
            counts.values() * float(w_syn_mv), size=counts.shape,
        ).to(device)
        self.register_buffer("W", weights)
        self.register_buffer("W_T", weights.transpose(0, 1).to_sparse_csr())
        # The signed synapse counts, kept unscaled. Every control arm in the
        # campaign is built from counts -- `specificity_campaign.substrate_vnc`
        # states it as a contract -- and recovering them by dividing the mV
        # matrix back down is not bit-exact, which the substrate cache checks
        # with `np.array_equal` against the frozen real VNC. So they are kept,
        # not reconstructed. `ShiuLIFStepper.W` is the accessor.
        self.register_buffer("W_counts", counts.to(device))
        self.w_syn_mv = float(w_syn_mv)
        self.register_buffer("descending_in", torch.from_numpy(io["descending_in"].astype(np.int64)).to(device))
        self.register_buffer("motor_out", torch.from_numpy(io["motor_out"].astype(np.int64)).to(device))

        self.dt = float(dt_ms)
        self.substeps = int(round(control_dt_ms / dt_ms))
        if abs(self.substeps * dt_ms - control_dt_ms) > 1e-9:
            raise ValueError("control_dt_ms must be an integer multiple of dt_ms")
        self.surrogate_slope = float(surrogate_slope)
        # Whether the gradient is allowed through the recurrent spike chain. The
        # forward pass is identical either way -- this only decides how far back
        # credit is assigned. See the note on the explosion in `forward`.
        self.detach_recurrent = bool(detach_recurrent)
        # Whether the reset carries gradient. It must not, and this is a real
        # implementation detail rather than a preference: with the reset attached,
        #     u <- u * (1 - s(u))
        # contributes  d/du = (1 - s) - u * ds/du  to the Jacobian, and u sits at
        # ~7 mV when a neuron spikes while ds/du peaks at slope/4 = 0.5, so every
        # spike multiplies the backward pass by roughly -3.5 BEFORE any recurrence
        # is involved. The forward pass is identical either way -- the reset still
        # happens -- so detaching costs nothing and removes a term that is an
        # artefact of the surrogate, not a derivative of the model. Every
        # reference SNN implementation does this (spikingjelly's `detach_reset`,
        # Zenke and Ganguli's SuperSpike).
        self.detach_reset = bool(detach_reset)
        self.w_syn = float(w_syn_mv)
        self.shuffle = False
        self.model = "shiu_lif"

        # Exponential-Euler coefficients. For du/dt = (-u + g)/T_MBR with
        # g(t) = g0 exp(-t/TAU), the exact step is
        #   u <- u * exp(-dt/T_MBR) + g0 * (TAU/(TAU - T_MBR)) * (exp(-dt/TAU) - exp(-dt/T_MBR))
        decay_u = math.exp(-self.dt / T_MBR)
        decay_g = math.exp(-self.dt / TAU)
        coupling = (TAU / (TAU - T_MBR)) * (decay_g - decay_u)
        self.register_buffer("decay_u", torch.tensor(decay_u, device=device))
        self.register_buffer("decay_g", torch.tensor(decay_g, device=device))
        self.register_buffer("coupling", torch.tensor(coupling, device=device))
        self.refractory_steps = float(math.ceil(T_RFC / self.dt))
        self.rate_ceiling_hz = 1000.0 / (self.refractory_steps * self.dt)
        # Mean of [SH]'s Poisson drive: rate (Hz) * dt (s) * w_syn * f_poi (mV).
        self.drive_mv_per_hz = self.w_syn * F_POI * (self.dt / 1000.0)
        # [SH] `model.py:85-92` -- a stimulated neuron gets a `PoissonInput` onto
        # `v` of weight `w_syn * f_poi` = 68.75 mV, and `neu[i].rfc = 0 * ms`.
        # Both halves matter and they only make sense together: one event is ten
        # times the 7 mV threshold, and nothing holds the neuron out afterwards,
        # so a driven neuron emits exactly one spike per Poisson event and its
        # firing rate IS the requested rate. It is a source, not an integrator.
        #
        # Substituting the Poisson process by its mean current -- what this core
        # did -- does not preserve that. A 100 Hz request becomes 13.75 mV every
        # substep, permanently over threshold, so the neuron fires on every
        # substep no matter what was asked: measured 341 / 489 / 498 Hz for
        # requests of 50 / 100 / 150 Hz. The rate was saturated, not encoded, and
        # the old whole-population refractory hid it by clamping the result to
        # 167 Hz -- close enough to look plausible, and wrong for both reasons.
        #
        # So the drive is restored as a spike source: one Bernoulli draw per
        # substep at p = rate * dt, which is the Poisson process thinned to the
        # substep grid (exact to O(p^2)). Forward is a real spike; backward is
        # straight-through to p, so the encoder still gets a gradient in the rate
        # it asked for. `descending_in` and `motor_out` are disjoint, so this
        # touches the drive and nothing about how the readout is bounded.
        self.drive_spike_prob_per_hz = self.dt / 1000.0
        # [SH]'s input is stochastic and so is this. Its own generator, so the
        # draw does not depend on -- or perturb -- global RNG state that the
        # rollout, the env and the controls all share.
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))
        is_drive = torch.zeros((self.n, 1), device=device)
        is_drive[self.descending_in] = 1.0
        self.register_buffer("refractory_per_neuron",
                             self.refractory_steps * (1.0 - is_drive))

        self.W.requires_grad_(False)
        self.eval()
        assert not tuple(self.parameters()), "the core must hold no trainable parameters"

    # -- state ---------------------------------------------------------------
    def initial_state(self, batch_size: int | None = None) -> torch.Tensor:
        if batch_size is not None and int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        shape = (self.state_dim,) if batch_size is None else (self.state_dim, int(batch_size))
        return torch.zeros(shape, dtype=torch.float32, device=self.W.device)

    def _unpack(self, state: torch.Tensor):
        return torch.split(state, self.n, dim=0)

    # -- dynamics ------------------------------------------------------------
    def _spike(self, u: torch.Tensor) -> torch.Tensor:
        """Heaviside forward, sigmoid surrogate backward.

        The slope governs ONLY the backward pass -- the forward spike is a hard
        threshold, so the firing rate is bit-identical at every slope (measured:
        11.54 Hz at 12 control steps for every value below). It is a property of
        the training method, not of [SH]'s model, and nothing about the published
        dynamics moves when it changes.

        It has to be far flatter here than the 10-25 (in threshold units) typical
        of trained SNNs, because those networks carry weights scaled by
        1/sqrt(N) while ours carries measured synapse counts -- up to 864 per
        edge. The chained Jacobian over a 24-step window is correspondingly
        steeper, so the surrogate has to be correspondingly flatter. Measured
        gradient norm reaching the drive, at 12 and 24 control steps:

            slope   @12        @24
            0.03    1487       2.58e07
            0.01    0.958      1.254
            0.003   0.202      0.204

        0.01 is the value that is bounded at the full window without vanishing.
        In threshold units (U_TH = 7 mV) it is 0.07.

        This was 0.03 while the stimulated neurons were driven by a mean current.
        Restoring [SH]'s Poisson source put a straight-through estimator on the
        input layer, whose local gradient is 1 rather than the surrogate's, and
        the chained window got steeper by roughly the same factor. The motor
        output is bit-identical across every slope in the table -- 0.02846 at 24
        steps for all three -- so nothing about the published dynamics moved.
        """
        surrogate = torch.sigmoid((u - U_TH) * self.surrogate_slope)
        return surrogate + ((u > U_TH).to(u.dtype) - surrogate).detach()

    def forward(self, descending_current: torch.Tensor, node_state: torch.Tensor):
        device = self.W.device
        current = torch.as_tensor(descending_current, dtype=torch.float32, device=device)
        state = torch.as_tensor(node_state, dtype=torch.float32, device=device)

        if state.ndim == 1:
            if state.shape[0] != self.state_dim or current.ndim != 1:
                raise ValueError(f"single node_state must be ({self.state_dim},)")
            batched, batch_size = False, None
            state = state[:, None]
        elif state.ndim == 2:
            if state.shape[0] != self.state_dim or current.ndim != 2 or current.shape[1] != state.shape[1]:
                raise ValueError(f"batched node_state must be ({self.state_dim}, B)")
            batched, batch_size = True, state.shape[1]
        else:
            raise ValueError(f"node_state has shape {tuple(state.shape)}")

        width = batch_size if batched else 1
        if current.ndim == 1:
            current = current[:, None]
        if current.shape[0] == len(self.descending_in):
            scattered = torch.zeros((self.n, width), dtype=current.dtype, device=device)
            current = scattered.index_copy(0, self.descending_in, current)
        elif current.shape[0] != self.n:
            raise ValueError(
                f"descending current has {current.shape[0]} rows; expected "
                f"{self.n} or {len(self.descending_in)}"
            )

        # The encoder's output is a stimulation RATE in Hz, exactly what [SH]
        # hands their Poisson input. Carried through as a per-substep spike
        # probability on the stimulated rows only; `_run_substeps` turns it into
        # the spikes of those neurons rather than a current into them. Clamped at
        # one spike per substep -- the resolution of the grid, not a model bound.
        drive = torch.clamp(
            current.index_select(0, self.descending_in) * self.drive_spike_prob_per_hz,
            0.0, 1.0)
        # Drawn HERE, outside the checkpoint. `torch.utils.checkpoint` reruns the
        # forward pass to rebuild the graph, and a generator advanced inside it
        # would hand the backward pass a different sample than the one the
        # forward actually used -- a silently wrong gradient. Only the stimulated
        # rows are drawn, so this is (substeps, 1325, B), not (substeps, n, B).
        noise = torch.rand((self.substeps,) + tuple(drive.shape),
                           generator=self.generator, device=device)

        u, g, spikes, refractory = self._unpack(state)
        if torch.is_grad_enabled():
            # Recompute the substeps in the backward pass instead of storing
            # them. The rate core holds ONE state tensor across 4 substeps; this
            # one holds four across 10, and each substep also pins the sparse
            # matmul output and the surrogate's intermediates. Un-checkpointed
            # that OOM-killed a 62 GB pod during the pre-flight suite, while
            # passing on a laptop -- the pod runs 27 OMP threads to the laptop's
            # 10, and the per-thread buffers scale with it. Checkpointing here
            # rather than shrinking the minibatch keeps PPO's arithmetic and the
            # comparison against the rate arm exactly as they were.
            outputs = torch.utils.checkpoint.checkpoint(
                self._run_substeps, drive, noise, u, g, spikes, refractory,
                use_reentrant=False,
            )
        else:
            outputs = self._run_substeps(drive, noise, u, g, spikes, refractory)
        u, g, spikes, refractory, spike_count = outputs

        duration_s = self.substeps * self.dt / 1000.0
        rate_hz = spike_count / duration_s
        # Normalised to the refractory ceiling so the decoder sees the same 0..1
        # scale the rate core's `value / fr_cap` gave it.
        motor = (rate_hz / self.rate_ceiling_hz)[self.motor_out]

        new_state = torch.cat((u, g, spikes, refractory), dim=0)
        if not batched:
            return new_state[:, 0], motor[:, 0]
        return new_state, motor

    def _run_substeps(self, drive, noise, u, g, spikes, refractory):
        from embody_policy import _FixedSparseMM  # local: avoids an import cycle

        spike_count = torch.zeros_like(u)
        for substep in range(self.substeps):
            # One substep of synaptic delay: this substep integrates the spikes
            # emitted on the last one.
            recurrent_input = spikes.detach() if self.detach_recurrent else spikes
            free = (refractory <= 0).to(u.dtype)
            # [SH] `model.py:46` marks `dg/dt` `(unless refractory)`: a refractory
            # neuron does not integrate g, but brian2 still applies `on_pre: g += w`
            # to it, so incoming synapses accumulate undecayed until it is free.
            g = free * (g * self.decay_g) + (1.0 - free) * g \
                + _FixedSparseMM.apply(self.W, self.W_T, recurrent_input)
            stepped = u * self.decay_u + g * self.coupling
            # A refractory neuron holds its reset potential and ignores input.
            u = free * stepped + (1.0 - free) * u
            spikes = self._spike(u) * free
            # A stimulated neuron keeps its own membrane dynamics -- [SH] add the
            # PoissonInput alongside its synapses, they do not replace them -- so
            # it spikes if either its membrane crosses OR an event arrives. One
            # Bernoulli draw per substep at p = rate * dt, straight-through to p
            # on the backward pass so the encoder still has a gradient in the
            # rate. `refractory_per_neuron` is zero here, so a draw always emits.
            source = drive + ((noise[substep] < drive).to(drive.dtype) - drive).detach()
            own = spikes.index_select(0, self.descending_in)
            spikes = spikes.index_copy(0, self.descending_in, own + source * (1.0 - own))
            spike_count = spike_count + spikes
            # Reset clears both the membrane and the synaptic accumulator, as in
            # [SH]'s `eq_rst`: 'v = v_rst; w = 0; g = 0 * mV'. See `detach_reset`.
            reset = spikes.detach() if self.detach_reset else spikes
            u = u * (1.0 - reset)
            g = g * (1.0 - reset)
            # The refractory counter is a clock, not a signal. Nothing downstream
            # reads it differentiably -- `free` is a comparison -- so keeping it in
            # the graph only pins 240 substeps of history in memory.
            refractory = (torch.clamp(refractory - 1.0, min=0.0)
                          + self.refractory_per_neuron * spikes.detach())

        return u, g, spikes, refractory, spike_count


class ShiuLIFStepper:
    """`VNCStepper`'s API, backed by [SH] dynamics.

    The zero-training audits (E0 phase, E1 turning, H4 dose-response, ...) all
    drive a `VNCStepper` directly rather than going through `EmbodyPolicy`, so
    this is what lets them ask the same questions of the published dynamics
    without touching their analysis code. One control step here is
    `substeps * dt` = 20 ms, matching the rate model's `substeps * dt` = 4 * 5 ms,
    so a history sampled from either is on the same time base.

    The drive argument keeps its meaning across the swap by luck of units: the
    audits drive at 100.0, and for this core that reads as 100 Hz of stimulation,
    inside the 150 Hz [SH] themselves use.
    """

    def __init__(self, data, weights_name: str = "vnc_weight_csr.pkl",
                 index_name: str = "vnc_io_index.npz", device: str = "cpu",
                 shuffle: bool = False, seed: int = 1,
                 noise_seed: int | None = None, **overrides) -> None:
        # Shuffling is refused inside ShiuLIFVNC so a training run cannot pick up
        # a scrambled graph by accident. A CONTROL arm needs exactly that, so it
        # is built here, deliberately, by the same column permutation VNCStepper
        # uses -- degree sequence and weight multiset preserved, wiring destroyed.
        frozen_weight = None
        if shuffle:
            import pickle

            with (Path(data) / weights_name).open("rb") as handle:
                base = pickle.load(handle)
            base = base.to_sparse_csr() if base.layout != torch.sparse_csr else base
            generator = torch.Generator().manual_seed(int(seed))
            permutation = torch.randperm(base.shape[1], generator=generator)
            frozen_weight = torch.sparse_csr_tensor(
                base.crow_indices(), permutation[base.col_indices()],
                base.values(), size=base.shape,
            )
        # The core's Poisson source is stochastic, so its seed selects a
        # REALIZATION, not just a shuffle. It was not forwarded, which pinned
        # every stepper to generator seed 1 and made the noise invisible --
        # `noise_seed` exists so a caller can average over realizations the way
        # [SH] do (`default_params` n_run = 30) instead of reporting one draw.
        overrides.setdefault("seed", int(noise_seed if noise_seed is not None else seed))
        self.core = ShiuLIFVNC(data=Path(data), weights_name=weights_name,
                               index_name=index_name, device=device,
                               frozen_weight=frozen_weight, **overrides)
        self.io = np.load(Path(data) / index_name)
        self.n = self.core.n
        self.device = self.core.W.device
        self.descending_in = self.io["descending_in"].astype(np.int64)
        self.motor_out = self.io["motor_out"].astype(np.int64)
        self.shuffle = bool(shuffle)
        self.model = "shiu_lif"
        self.substeps = self.core.substeps
        self.dt = self.core.dt / 1000.0
        self.seed = int(seed)
        self.noise_seed = int(noise_seed if noise_seed is not None else seed)
        self.reset()

    # `VNCStepper` exposes the graph as `.W` in signed synapse counts, and the
    # audits read and write it by that name: `build_vnc_substrates` reads it to
    # derive the rewired and randomised controls, `audit_h5_lesion_localization`
    # reads it to lesion. Without this the whole campaign died on
    # `'ShiuLIFStepper' object has no attribute 'W'` the moment SC_CORE=lif, so
    # E1 / E3 / H1 / H4 / H5 had never once run under the published dynamics.
    # Counts in, counts out -- the mV scaling is this core's business alone.
    @property
    def W(self):
        return self.core.W_counts

    @W.setter
    def W(self, counts):
        counts = counts.to_sparse_csr() if counts.layout != torch.sparse_csr else counts
        # Control substrates are built CPU-side (the rewire routines are numpy),
        # so a graph handed in here has to be moved onto whatever device the core
        # is running on -- otherwise the first sparse matmul dies on
        # "found at least two devices, cpu and cuda:0".
        device = self.core.descending_in.device
        counts = counts.to(device)
        scaled = torch.sparse_csr_tensor(
            counts.crow_indices(), counts.col_indices(),
            counts.values().float() * self.core.w_syn_mv, size=counts.shape,
        )
        self.core.W_counts = counts
        self.core.W = scaled
        self.core.W_T = scaled.transpose(0, 1).to_sparse_csr()
        self.reset()

    def reset(self) -> None:
        self._state = self.core.initial_state()

    # Identical to VNCStepper.descending_current; the audits call it by name.
    def descending_current(self, drive, turn=0.0, dnb08_gain=1.0, turn_scale=1.0):
        drive = max(0.0, float(drive))
        bias = float(np.clip(float(turn) / max(float(turn_scale), 1e-6), -1.0, 1.0))
        left = drive * (1.0 - bias)
        right = drive * (1.0 + bias)
        current = np.zeros(self.n, dtype=np.float32)
        for key, value in (("dng100_left", left), ("dng100_right", right)):
            idx = self.io[key]
            if len(idx):
                current[idx] = value
        for key, value in (("dnb08_left", left * dnb08_gain), ("dnb08_right", right * dnb08_gain)):
            idx = self.io[key]
            if len(idx):
                current[idx] = value
        return current

    def step(self, I_descending):
        """Advance 20 ms and return normalised rates at `motor_out`."""
        x = torch.as_tensor(I_descending, dtype=torch.float32, device=self.device).flatten()
        if x.numel() == self.n:
            current = x
        elif x.numel() == len(self.descending_in):
            current = torch.zeros(self.n, dtype=torch.float32, device=self.device)
            current[torch.from_numpy(self.descending_in).to(self.device)] = x
        else:
            raise ValueError(
                f"descending current has {x.numel()} values; expected {self.n} "
                f"or {len(self.descending_in)}"
            )
        with torch.no_grad():
            self._state, motor = self.core(current, self._state)
        return motor.detach()


def demo() -> None:
    """Self-check against the published model's own guarantees."""
    here = Path(__file__).resolve()
    # The graph lives under `data/` in the repo and under `results/path2/` on a
    # deployed pod, which carries only the leg subgraph. The self-check has to
    # run in both, because a pod that skips it runs the whole campaign on
    # whatever the core happens to do.
    root = here.parent.parent
    if (root / "data" / "vnc_weight_csr.pkl").exists():
        data, weights_name, index_name = root / "data", "vnc_weight_csr.pkl", "vnc_io_index.npz"
    elif (root / "results" / "path2" / "all_legs_weight_csr.pkl").exists():
        data = root / "results" / "path2"
        weights_name, index_name = "all_legs_weight_csr.pkl", "all_legs_io_index.npz"
    else:
        raise SystemExit(f"no frozen VNC graph under {root}/data or {root}/results/path2")
    names = {"weights_name": weights_name, "index_name": index_name}
    core = ShiuLIFVNC(data=data, **names)
    assert not tuple(core.parameters())
    assert core.state_dim == STATE_BLOCKS * core.n

    n_drive = len(core.descending_in)
    batch = 2

    # 1. [SH] give their neurons zero basal firing rate. Silence in, silence out.
    state = core.initial_state(batch)
    new_state, motor = core(torch.zeros(n_drive, batch), state)
    assert float(motor.abs().max()) == 0.0, "the unstimulated network spiked"
    assert float(new_state.abs().max()) == 0.0

    # 2. Driven, it fires -- and never faster than the refractory period allows.
    drive = torch.full((n_drive, batch), R_POI_MAX)
    state = core.initial_state(batch)
    for _ in range(10):
        state, motor = core(drive, state)
    assert float(motor.max()) > 0.0, "stimulating every descending neuron produced no motor output"
    assert float(motor.max()) <= 1.0 + 1e-6, f"rate above the refractory ceiling: {float(motor.max())}"

    # 2b. [SH] strip the refractory period from their Poisson targets, and one
    #     reference event clears threshold on its own, so a driven neuron fires at
    #     the rate it was asked for. Ours must too, or the encoder's output is not
    #     the quantity [SH] hand their Poisson input. Checked against the ceiling
    #     the discrete integrator actually enforces, not the continuous-time one.
    n_control = 40
    seconds = n_control * core.substeps * core.dt / 1000.0
    for asked in (50.0, 100.0, R_POI_MAX):
        u, g, s, rfc = [t.clone() for t in torch.split(core.initial_state(1), core.n, dim=0)]
        p = torch.full((n_drive, 1), asked * core.drive_spike_prob_per_hz)
        counted = torch.zeros((core.n, 1))
        with torch.no_grad():
            for _ in range(n_control):
                noise = torch.rand((core.substeps,) + tuple(p.shape), generator=core.generator)
                u, g, s, rfc, sc = core._run_substeps(p, noise, u, g, s, rfc)
                counted += sc
        seen = float((counted / seconds)[core.descending_in].mean())
        # In the wired network a stimulated neuron also spikes on its recurrent
        # input, exactly as [SH]'s does, so the rate can only sit above what was
        # asked. The equality is checked without recurrence just below.
        assert asked <= seen < 1.6 * asked, f"asked {asked} Hz of drive, got {seen:.1f} Hz"

    # With the graph silenced there is nothing but the Poisson source left, and
    # then the stimulated rate must BE the requested rate. This is the assertion
    # the mean-current drive failed -- it returned 341 Hz for a 50 Hz request.
    silent = ShiuLIFVNC(data=data, **names,
                        frozen_weight=torch.zeros(core.n, core.n).to_sparse_csr())
    for asked in (50.0, 100.0, R_POI_MAX):
        u, g, s, rfc = [t.clone() for t in torch.split(silent.initial_state(1), silent.n, dim=0)]
        p = torch.full((n_drive, 1), asked * silent.drive_spike_prob_per_hz)
        counted = torch.zeros((silent.n, 1))
        with torch.no_grad():
            for _ in range(n_control):
                noise = torch.rand((silent.substeps,) + tuple(p.shape), generator=silent.generator)
                u, g, s, rfc, sc = silent._run_substeps(p, noise, u, g, s, rfc)
                counted += sc
        seen = float((counted / seconds)[silent.descending_in].mean())
        assert abs(seen - asked) < 0.05 * asked, f"unwired: asked {asked} Hz, got {seen:.1f} Hz"
    # The readout's divisor must be the ceiling the integrator enforces, and the
    # motor neurons -- unlike the stimulated ones -- must still carry a refractory.
    assert core.rate_ceiling_hz <= RATE_CEILING_HZ + 1e-9
    assert core.rate_ceiling_hz == 1000.0 / (core.refractory_steps * core.dt)
    assert float(core.refractory_per_neuron[core.motor_out].min()) > 0.0
    assert float(core.refractory_per_neuron[core.descending_in].max()) == 0.0

    # 3. The exponential-Euler step is the exact solution, not an approximation.
    u0, g0 = 3.0, 2.0
    coarse = u0 * core.decay_u.item() + g0 * core.coupling.item()
    u, g, micro = u0, g0, 4000
    h = core.dt / micro
    for _ in range(micro):
        u, g = u + h * (-u + g) / T_MBR, g - h * g / TAU
    assert abs(coarse - u) < 1e-3, f"integrator mismatch: {coarse} vs {u}"

    # 4. The gradient PPO needs reaches the drive through the spikes.
    drive = torch.zeros(n_drive, 1, requires_grad=True)
    with torch.no_grad():
        drive += R_POI_MAX
    state = core.initial_state(1)
    for _ in range(3):
        state, motor = core(drive, state)
        state = state.detach()
    motor.sum().backward()
    grad = float(drive.grad.abs().sum())
    assert math.isfinite(grad) and grad > 0.0, f"surrogate gradient is dead: {grad}"

    # 4b. The gradient survives a FULL PPO window, and the surrogate slope does
    #     not touch the forward pass. Both halves matter: an attached reset made
    #     the gradient explode here (2.6e15 at 12 steps) and made the slope sweep
    #     non-monotone, which read as a property of the model rather than a bug.
    n_control = 24
    rates_by_slope = {}
    for slope in (0.01, 2.0):
        probe = ShiuLIFVNC(data=data, surrogate_slope=slope, **names)
        drive = torch.zeros(n_drive, 2, requires_grad=True)
        with torch.no_grad():
            drive += 0.5 * R_POI_MAX
        state = probe.initial_state(2)
        for _ in range(n_control):
            state, motor = probe(drive, state)
        rates_by_slope[slope] = float(motor.detach().mean())
        if slope == probe.surrogate_slope == 0.01:
            motor.sum().backward()
            norm = float(drive.grad.norm())
            assert math.isfinite(norm), f"gradient is not finite over {n_control} steps"
            assert norm < 1.0e3, f"gradient {norm:.4g} over {n_control} steps will not train"
    assert rates_by_slope[0.01] == rates_by_slope[2.0], (
        f"the surrogate slope changed the forward pass: {rates_by_slope}; it must "
        "only affect the backward pass"
    )

    # 5. Batched and single paths are the same arithmetic, as FrozenVNC requires.
    #    Driven at a rate whose per-substep probability saturates at 1, so every
    #    draw fires and the two paths are comparable at all: the Poisson source is
    #    stochastic, as [SH]'s is, and its samples are independent across the
    #    batch by design. Everything downstream of the source is deterministic,
    #    which is what this check is actually about.
    single_state = core.initial_state()
    single_drive = torch.full((n_drive,), 1.0 / core.drive_spike_prob_per_hz)
    single_state, single_motor = core(single_drive, single_state)
    pair_state, pair_motor = core(
        torch.stack([single_drive, single_drive], dim=1), core.initial_state(2)
    )
    delta = float((pair_motor[:, 0] - single_motor).abs().max())
    assert delta <= 1e-5, f"batched core differs from the single path by {delta:.3g}"

    # 6. The measured graph is unchanged: signed synapse counts times w_syn.
    import pickle
    raw = pickle.load((data / weights_name).open("rb"))
    raw = raw.to_sparse_csr() if raw.layout != torch.sparse_csr else raw
    scaled = raw.values().float() * W_SYN
    assert torch.allclose(core.W.values(), scaled), "the core is not the measured matrix"

    print(f"vnc_lif_core: all checks passed (n={core.n} substeps={core.substeps} "
          f"dt={core.dt}ms w_syn={core.w_syn}mV)")


if __name__ == "__main__":
    demo()
