# Research scout findings

**Date:** 2026-08-31
**Purpose:** decide whether `EMBODY_3WAY` should run as registered, and how to
frame it. This is reconnaissance, not a paper. No experiment code or training
was changed.

**Reading key:** `Read` means the full article or relevant methods/results were
inspected. `Skimmed` means abstract, metadata, or searchable methods were
inspected. Recent preprints are evidence to check, not field consensus.

## Local setup

The registered experiment is a frozen 11,000-node MANC VNC rate core with a
linear encoder from hexapod observations into **all 1,325 descending inputs**
and a linear decoder into 18 joints. Stage 1 is complete; Stage 2 is a
one-environment PPO run capped at 5,000,000 steps per arm, with 12 evaluation
seeds, and has not started. The current CPU implementation is the only working
path because CUDA is absent and MPS fails at sparse-CSR construction. [L1][L2][L3]

## Decision summary

- **Do not call the broad idea novel.** Frozen connectome reservoirs and
  connectome-driven non-animal robots already exist. [R1][R4][R5]
- **A narrower intersection remains plausible:** measured Drosophila MANC/VNC
  weights, a non-fly 18-DoF hexapod, a fixed core, and explicit structural
  controls. I found no located source combining all four.
- **Recommended encoder width: 4 latent command dimensions**, expanded into
  descending inputs by a fixed, predeclared map. Four is an engineering
  bottleneck, not a measured biological control-rank estimate. [R9][R10][R11]
- **Do not run Stage 2 unchanged.** First decide whether to replace the
  1,325 independently learned DN drives with that bottleneck and whether the
  scientific endpoint is synthetic locomotion transfer or biological gait
  transfer.

## Q1 - Has anyone driven a non-fly body with a frozen measured connectome?

**Yes in the broad category; not found for the exact fly-VNC-to-hexapod
intersection.** Valencia Urbina, Cannas, and Gleiser used a numerical
*C. elegans* connectome model to drive GoPiGo and custom two-wheel robots; the
paper describes connectome-derived synaptic weights and common neuron
thresholds, with no learned interface reported. That already occupies
“measured connectome controlling a non-animal body.” Portegys is the important
counterexample: its artificial-worm study trains synapse weights, so
“connectome robot” does not automatically mean frozen. FlyGM uses a whole
Drosophila connectome but a biomechanical Drosophila body and trains an encoder,
graph update, intrinsic descriptors, decoder, imitation stage, and PPO. The
safe claim is therefore “no located precedent for frozen measured MANC/VNC
weights controlling a non-fly 18-DoF hexapod under this control design,” not
“first connectome robot.” [R1][R2][R3]

## Q2 - Is frozen biological recurrent core plus thin trained interface new?

**No as an architecture family; possibly yes as this embodied negative.**
BioESN work fixes an empirical connectome-derived reservoir topology and trains
the readout, while `conn2res` explicitly fixes the reservoir architecture and
local dynamics and trains a linear readout. Morra and Daley apply Drosophila
connectome topology to an ESN, and Dold, Hale, and Izzo use Drosophila topology
and synaptic connectivity for time-series prediction. These precedents make
“fixed recurrent biological structure plus learned thin output” insufficient as
the novelty claim. They do not test a measured MANC VNC rate core in a
different legged body. I found no published result proving that interface
training always erases measured connectivity; the 2026 Dhiman preprint is the
closest warning that weak nulls can create a false connectivity advantage, while
shared initialization and degree-preserving nulls remove it. [R4][R5][R6][R7][R8]

## Q3 - How wide should the descending encoder be?

**The literature supports a bottleneck, but does not support a biological
number of 2-5.** Aymanns, Chen, and Ramdya describe roughly 1,100 fly DNs,
find many walk-encoding neurons and strong turning representation, and find
far fewer DNs weakly encoding speed. Bidaye et al. identify distinct P9 and BPN
walking-initiation pathways, while Rayshubskiy et al. identify separate
low-gain and high-gain steering channels with a left-right “see-saw” command.
Those findings argue against both a single scalar and 1,325 independently
learned drives, but population PCA variance is not the same thing as causal
control rank. **Recommendation: 4 latent commands** - forward
initiation/drive, left-right steering, speed/arousal, and locomotor mode -
followed by a fixed, predeclared expansion into the 1,325 DN inputs. If the
expansion is learned per DN, the bottleneck is cosmetic and the core remains
puppetable. [R9][R10][R11][L2]

## Q4 - Are amplitude-matched random plus degree-preserving rewiring accepted?

**Yes as a strong first control; spatial/mesoscale controls are the main gap.**
The local protocol is better than a plain random baseline: its random arm
permutes signed weights on the same sparse graph, and its rewired arm preserves
each node’s in/out degree and the exact weight multiset. Lin et al.’s Drosophila
network analysis uses Erdős-Rényi, a directed configuration model preserving
degree sequences, and a more conservative neuropil/block-aware model. Váša and
Mišić describe nulls as a spectrum whose preserved properties must match the
hypothesis. Add a distance- or neuropil-preserving degree null before making a
fine-grained anatomical mechanism claim. I found no field convention matching
the local exclusion of **208/260** motor channels during amplitude matching;
that scale is a red flag and should be shown as a declared sensitivity analysis,
not treated as normal. [R4][R8][R13][R14][L5]

## Q5 - Is a synthetic body-path teacher defensible without target-morphology mocap?

**Defensible as task shaping; weaker than imitation and insufficient for a gait
claim.** Flybody and DeepMimic use recorded or reference motion when the target
body has such data. In contrast, Ouyang et al. use a low-dimensional CPG target
for an 18-servo hexapod, and Kim et al. use command-conditioned kinematic
trajectories as an imitation prior for an 18-DoF hexapod trained with PPO.
Thus a synthetic constant-speed path is a legitimate locomotion objective, but
it leaves contact sequence, limb phase, joint kinematics, energy, and
anti-sliding constraints underdetermined. Call Option B **synthetic
body-path pretraining** or a **task teacher**, not biological imitation; add
joint/contact/drag diagnostics before calling the result a transferred gait.
[R15][R16][R17][R18][L2]

## Q6 - What training and hardware choices are practical?

**Five million steps per arm is a reasonable pilot cap, not an emergence
guarantee.** Successful legged-PPO systems commonly use thousands of parallel
simulators: Rudin et al. demonstrate this for ANYmal, and the recent 18-DoF
hexapod PPO report uses 4,096 Isaac Gym agents. Those results do not provide a
portable single-environment budget or exact hyperparameters for this frozen
11,000-node rate core; no known-good 18-DoF single-env MuJoCo recipe emerged
from the search. The cheapest correct execution path is the existing CPU sparse
CSR core. CUDA is the preferred acceleration path if available. Dense fallback
should be a separately benchmarked implementation, not an automatic backend
switch: a dense 11,000 x 11,000 float32 matrix is roughly 484 MB before states,
optimizer memory, and copies. PyTorch’s official MPS documentation covers the
general backend, while its tracker records the same `new_compressed_tensor` MPS
CSR failure seen locally; I found no current official support path to rely on.
[R18][R19][R22][R23][L3]

## Q7 - What counts as morphology transfer?

**A single fitted hexapod is embodiment, not morphology generalization.** URMA
trains a shared abstract locomotion controller with morphology-agnostic
encoders and decoders and reports transfer to unseen robot platforms. The
neural-circuit-prior work likewise transfers a learned architecture to new body
designs, but neither uses a frozen measured MANC core. To claim transfer across
bodies, keep the recurrent core and latent command semantics fixed, evaluate a
held-out body without fitting a new per-body encoder/decoder, or explicitly
label the result as one-body adaptation. The present registered design supports
the latter claim only. [R20][R21][L2]

## Evidence table

| Source | Frozen | Trained | Body/task | Controls | Relevance |
|---|---|---|---|---|---|
| C. elegans robot [R1] (Read) | Connectome weights and simple neuron thresholds | No learned interface reported | GoPiGo/custom two-wheel robot | Randomized degree/weight distributions | Broad measured-connectome-to-non-animal precedent; not a legged body |
| Artificial C. elegans [R2] (Skimmed) | Connectome starting structure | Synapse weights fitted by genetic algorithm | Artificial-worm sensorimotor task | Not the present structural-null design | Shows connectome robotics often tunes the circuit |
| FlyGM [R3] (Skimmed) | Connectome operator/topology | Encoder, gate, intrinsic descriptors, graph update, decoder, IL, PPO | Biomechanical Drosophila | Random graph, degree-preserving rewiring, MLP | Closest Drosophila embodied precedent, but learned core and fly body |
| BioESN [R4] (Read) | Empirical topology; ESN weights are initialized/varied | Linear readout | Memory tasks | Bio rank/no-rank and random surrogates | Establishes connectome-as-reservoir control logic |
| `conn2res` [R5] (Skimmed) | Connectome architecture and local dynamics | Linear readout | Cognitive/time-series tasks | Configurable rewiring and dynamics | Exact thin-readout seam, no embodiment |
| Drosophila ESN [R6][R7] (Skimmed) | Fly topology; Dold et al. also retain synaptic connectivity | Readout | Chaotic time series | ESN/random comparisons | Makes topology/weights-plus-readout non-novel outside locomotion |
| Dhiman [R8] (Skimmed, preprint) | Drosophila graph in controlled comparison | Models trained from shared initialization | Flyvis study | Naive random vs degree-preserving null | Direct warning about weak controls and checkpoint leakage |
| DN population studies [R9][R10][R11] (Skimmed) | Biological measurements/connectivity, not a controller | None | Drosophila walking | Experimental perturbation and population analyses | Supports multiple command channels, not a 4-D ground truth |
| Fly network nulls [R13][R14] (Skimmed) | Empirical network for analysis | None | Drosophila connectome statistics | ER, degree/configuration, neuropil/block-aware nulls | Supports stronger spatial/mesoscale controls |
| Flybody [R15] (Skimmed) | Body model and recorded motion targets | MLP/RL controllers | Drosophila body | Real trajectory/motion datasets | Reference for biological motion targets |
| DeepMimic [R16] (Skimmed) | Reference clips | Physics policy | Multiple simulated bodies | Task plus motion imitation objectives | Shows why body-path-only targets are weaker |
| Hexapod PPO [R17] (Skimmed) | Synthetic kinematic guidance, not connectome | PPO torque policy | 18-DoF hexapod, 4,096 parallel agents | Position baseline and annealed prior | Direct practical precedent for synthetic teacher + PPO |
| Hexapod CPG-RL [R18] (Skimmed) | CPG structure | Two CPG parameters via RL | 18-servo hexapod | Terrain and gait comparisons | Low-dimensional task shaping precedent; not frozen connectome |
| Massively parallel PPO [R19] (Skimmed) | No biological core | PPO policy | ANYmal quadruped | Thousands of parallel simulators | Warns that single-env 5M steps may be throughput-limited |
| URMA [R20] (Skimmed) | No biological core | Shared controller plus morphology encoders/decoders | Multiple and unseen legged morphologies | Multi-embodiment training | Defines stronger evidence for morphology transfer |

## Novelty assessment

**Already established:** connectome-derived reservoirs, fixed recurrent dynamics
with trained readouts, and degree-preserving structural nulls. The broad claim
“a biological connectome can drive a robot” is also already occupied by the
*C. elegans* robot work. [R1][R4][R5][R13]

**Still potentially differentiating:** measured Drosophila MANC/VNC synaptic
weights used as a fixed rate core; a non-fly 18-DoF hexapod; a fixed anatomical
map with a genuinely narrow interface; and a predeclared comparison against
weight-matched and degree-preserving controls. I found no located source with
that full intersection, but this was targeted scouting, not an exhaustive
systematic review. [L1][L2][L5]

**Not currently supportable:** “first,” “connectome structure is irrelevant
after interface training,” or any Stage 2 outcome. The local EMBODY report says
Stage 1 only; its action MSE is not locomotion and has no PPO displacement,
confidence interval, or decision label. [L3]

## Ranked changes before Stage 2

1. **Replace the 1,325-output encoder with a 4-D latent command bottleneck.**
   Expand it through a fixed, versioned DN map. Do not learn one independent
   gain/bias per DN if the goal is to test the core. Four is a design choice
   motivated by distinct forward and steering channels, not a claimed biological
   dimensionality. [R9][R10][R11][L2]
2. **Keep the current weight/degree controls; add spatial structure only for a
   stronger mechanism claim.** Preserve the current exact weight multiset and
   degree logic. Add neuropil/distance-preserving rewires if the conclusion is
   about anatomical organization rather than this particular null. [R13][R14][L5]
3. **Expose all 260 motor channels as a sensitivity result.** The 208-channel
   exclusion is too large to hide behind amplitude matching. Report the common
   mask, excluded channels, per-channel SD error, and an all-channel analysis;
   do not retune constants after registration. [L3][L5]
4. **Rename Option B.** Use “synthetic body-path teacher” or “task-conditioned
   pretraining.” Require contact, phase, joint-range, and anti-drag diagnostics
   before using “tripod gait” or biological imitation. [R15][R16][R17][L2]
5. **Run CPU sparse or obtain CUDA before spending the full budget.** Keep all
   three arms on the same backend and budget. Treat 5M as a pilot; a null after
   a throughput-starved single-env run cannot establish that emergence was
   given a field-comparable opportunity. [R17][R19][R22][R23][L3]

## Threats a reviewer will raise

- The 1,325-channel learned encoder can directly program descending neurons;
  any later connectivity result may be an interface result. [L2][L5]
- A body-path teacher does not identify limb phase, contacts, energy, or a
  non-sliding gait. [R15][R16][R17][L2]
- Excluding 208/260 motor channels creates a selection/amplitude concern even
  when the matching rule is declared. [L3][L5]
- Degree and weight matching do not preserve spatial distance, neuropil
  boundaries, motifs, or cell-type/pathway roles. [R13][R14]
- The source nervous system and body are mismatched: MANC/VNC dynamics, a
  hand-specified hexapod map, and one body do not establish biological transfer
  or cross-morphology generalization. [R1][R3][R20][L2]
- A rate model omits biological delays, adaptation, detailed synaptic dynamics,
  and validated cell-specific physiology; a connectome is not a complete
  functional model. [R3][R9][L2]
- One CPU environment and 5M steps per arm may be a compute limitation rather
  than evidence for or against learned locomotion. [R17][R19][L3]
- Stage 1 loss differences cannot be presented as walking performance; Stage 2
  remains unrun. [L3]

## Open / unresolved

- No located paper combined frozen measured Drosophila MANC/VNC weights, a
  non-fly legged body, a trained thin interface, and the present null ladder.
  Search covered the three local fly reference repos, arXiv, PubMed/PMC,
  Nature, eLife, PLOS, NeurIPS, CoRL/PMLR, Frontiers, MDPI, and official
  PyTorch documentation/issues. This is not proof of nonexistence.
- No accepted quantitative DN control rank was found. DN population coding and
  PCA motivate a bottleneck but do not identify 4 as a biological fact. [R9]
- No published benchmark was found for an amplitude-match mask discarding
  208/260 outputs. Treat the count as a project-specific warning. [L3]
- No reliable single-environment PPO budget or hyperparameter set for this exact
  11,000-node sparse rate core was found. [R17][R19]
- No supported MPS sparse-CSR execution path was identified in current official
  PyTorch material. Verify installed-version behavior again only if changing
  backend; do not silently fall back during the registered run. [R22][R23]
- FlyGM and Dhiman are 2026 arXiv preprints, and the recent hexapod PPO source
  is 2026. Independent replication and peer-review status should be checked
  before making priority claims. [R3][R8][R17]

## Sources

### Local files

- **[L1] Read:** [`docs/archive/FIELD_CODE_REVIEW.md`](docs/archive/FIELD_CODE_REVIEW.md).
- **[L2] Read:** [`docs/archive/EMBODY_3WAY_SPEC.md`](docs/archive/EMBODY_3WAY_SPEC.md).
- **[L3] Read:** [`eon-fly-brain/results/path2/RUN_EMBODY_3WAY.md`](eon-fly-brain/results/path2/RUN_EMBODY_3WAY.md).
- **[L4] Read:** [`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md).
- **[L5] Read:** [`PATH2_METHODS.md`](PATH2_METHODS.md).

### External literature and technical sources

- **[R1] Read:** Carlos E. Valencia Urbina, Sergio A. Cannas, and Pablo M. Gleiser, “Emergent dynamics in a robotic model based on the *Caenorhabditis elegans* connectome,” *Frontiers in Neurorobotics* 16:1041410 (2023). [DOI](https://doi.org/10.3389/fnbot.2022.1041410).
- **[R2] Skimmed:** Thomas E. Portegys, “Training sensory-motor behavior in the connectome of an artificial *C. elegans*,” *Neurocomputing* 168, 128-134 (2015). [DOI](https://doi.org/10.1016/j.neucom.2015.06.007).
- **[R3] Skimmed:** Zehao Jin, Yaoye Zhu, Chen Zhang, and Yanan Sui, “Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly,” arXiv:2602.17997 (2026). [Paper](https://arxiv.org/abs/2602.17997).
- **[R4] Read:** Fabrizio Damicelli, Claus C. Hilgetag, and Alexandros Goulas, “Brain connectivity meets reservoir computing,” *PLOS Computational Biology* 18(11):e1010639 (2022). [DOI](https://doi.org/10.1371/journal.pcbi.1010639).
- **[R5] Skimmed:** Laura E. Suárez, Agoston Mihalik, Filip Milisav, Kenji Marshall, Mingze Li, Petra E. Vértes, Guillaume Lajoie, and Bratislav Mišić, “Connectome-based reservoir computing with the conn2res toolbox,” *Nature Communications* 15:656 (2024). [DOI](https://doi.org/10.1038/s41467-024-44900-4).
- **[R6] Skimmed:** Jacob Morra and Mark Daley, “Imposing Connectome-Derived Topology on an Echo State Network,” *Proceedings of IJCNN* (2022). [DOI](https://doi.org/10.1109/IJCNN55064.2022.9892629).
- **[R7] Skimmed:** Dominik Dold, Zachary F. Hale, and Dario Izzo, “The Drosophila Connectome as a Computational Reservoir for Time-Series Prediction,” *Biomimetics* 10(5):341 (2025). [DOI](https://doi.org/10.3390/biomimetics10050341).
- **[R8] Skimmed; preprint:** Nalin Dhiman, “Topological Sensitivity in Connectome-Constrained Neural Networks,” arXiv:2604.04033 (2026). [Paper](https://arxiv.org/abs/2604.04033).
- **[R9] Skimmed:** Florian Aymanns, Chin-Lin Chen, and Pavan Ramdya, “Descending neuron population dynamics during odor-evoked and spontaneous limb-dependent behaviors,” *eLife* 11:e81527 (2022). [DOI](https://doi.org/10.7554/eLife.81527).
- **[R10] Skimmed:** Salil S. Bidaye, Meghan Laturney, Amy K. Chang, Yuejiang Liu, Till Bockemühl, Ansgar Büschges, and Kristin Scott, “Two brain pathways initiate distinct forward walking programs in *Drosophila*,” *Neuron* 108, 469-485.e8 (2020). [DOI](https://doi.org/10.1016/j.neuron.2020.07.032).
- **[R11] Skimmed:** Aleksandr Rayshubskiy, Stephen L. Holtz, Alexander S. Bates, Quinn X. Vanderbeck, Laia Serratosa Capdevila, Victoria Rockwell, and Rachel I. Wilson, “Neural circuit mechanisms for steering control in walking *Drosophila*,” *eLife* 13:RP102230 (2025). [DOI](https://doi.org/10.7554/eLife.102230).
- **[R13] Skimmed:** Albert Lin, Runzhe Yang, Sven Dorkenwald, Arie Matsliah, Amy R. Sterling, Philipp Schlegel, Szi-chieh Yu, Claire E. McKellar, Marta Costa, Katharina Eichler, Alexander Shakeel Bates, Nils Eckstein, Jan Funke, Gregory S. X. E. Jefferis, and Mala Murthy, “Network statistics of the whole-brain connectome of *Drosophila*,” *Nature* 634, 153-165 (2024). [DOI](https://doi.org/10.1038/s41586-024-07968-y).
- **[R14] Skimmed:** František Váša and Bratislav Mišić, “Null models in network neuroscience,” *Nature Reviews Neuroscience* 23, 493-504 (2022). [DOI](https://doi.org/10.1038/s41583-022-00601-9).
- **[R15] Skimmed:** Roman Vaxenburg, Igor Siwanowicz, Josh Merel, Alice A. Robie, Carmen Morrow, Guido Novati, Zinovia Stefanidi, Gert-Jan Both, Gwyneth M. Card, Michael B. Reiser, Matthew M. Botvinick, Kristin M. Branson, Yuval Tassa, and Srinivas C. Turaga, “Whole-body physics simulation of fruit fly locomotion,” *Nature* 643, 1312-1320 (2025). [DOI](https://doi.org/10.1038/s41586-025-09029-4).
- **[R16] Skimmed:** Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel van de Panne, “DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills,” *ACM Transactions on Graphics / SIGGRAPH* (2018). [Paper](https://arxiv.org/abs/1804.02717).
- **[R17] Skimmed:** Taesoo Kim, Minjun Choi, Seunguk Choi, Taeuan Yoon, and Dongil Choi, “Deep Reinforcement Learning Control of a Hexapod Robot,” *Actuators* 15(1):33 (2026). [DOI](https://doi.org/10.3390/act15010033).
- **[R18] Skimmed:** Wenjuan Ouyang, Haozhen Chi, Jiangnan Pang, Wenyu Liang, and Qinyuan Ren, “Adaptive Locomotion Control of a Hexapod Robot via Bio-Inspired Learning,” *Frontiers in Neurorobotics* 15:627157 (2021). [DOI](https://doi.org/10.3389/fnbot.2021.627157).
- **[R19] Skimmed:** Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter, “Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning,” *Proceedings of the Conference on Robot Learning* 164, 91-100 (2022). [Paper](https://proceedings.mlr.press/v164/rudin22a.html).
- **[R20] Skimmed:** Nico Bohlinger, Grzegorz Czechmanowski, Maciej Piotr Krupka, Piotr Kicki, Krzysztof Walas, Jan Peters, and Davide Tateo, “One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion,” *Proceedings of the Conference on Robot Learning* 270, 3356-3378 (2025). [Paper](https://proceedings.mlr.press/v270/bohlinger25a.html).
- **[R21] Skimmed:** Nikhil X. Bhattasali, Anthony M. Zador, and Tatiana A. Engel, “Neural Circuit Architectural Priors for Embodied Control,” *Advances in Neural Information Processing Systems* 35 (2022). [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/52e431bd7689d98426300cb103bb0ee3-Abstract-Conference.html).
- **[R22] Read:** PyTorch, “MPS backend,” official documentation, current page updated 2026-03-13. [Documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).
- **[R23] Read:** PyTorch issue #129842, “MPS Sparse Support,” official tracker; records `new_compressed_tensor` failure for sparse CSR on MPS. [Issue](https://github.com/pytorch/pytorch/issues/129842).
