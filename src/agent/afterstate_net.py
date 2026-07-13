"""Neural afterstate value network for 2048 (Track C1) — the NN that *should* work.

Why this and not the DQN? The DQN learns ``Q(s,a)`` and fails because in 2048 the
action-value gaps are tiny next to the state value, so ``Q(s,a) ≈ V(s)`` and greedy
degenerates to random (see ``docs/dqn_report.md``). This network instead learns
``V(afterstate)`` — the value of the board right after a move's merges, before the
random spawn — and acts by ``argmax_a [reward(s,a) + V(afterstate(s,a))]``. That is
the *exact* principle that makes the tabular n-tuple strong; here the lookup tables
are replaced by a CNN, and everything runs batched on the GPU via
:class:`~src.game.model.vectorboard.TorchVectorEngine`.

Design choices that matter:
  * **One-hot exponent input** (16 channels) — same as the DQN; lets the net treat
    each tile magnitude as a distinct feature.
  * **Reward scaling** — merge rewards are divided by ``SCALE`` so the regression
    target ``V`` (a cumulative future score) stays O(10), not O(10^5). Scaling both
    reward and V identically leaves the argmax policy unchanged.
  * **Dihedral augmentation** — each training board gets a random one of the 8
    board symmetries, substituting for the n-tuple's symmetric weight sharing.
  * **Online afterstate TD(0)** on thousands of parallel games — the network sees a
    large, decorrelated batch of afterstates every gradient step.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game.model.vectorboard import TorchVectorEngine, DIRECTIONS

NUM_CHANNELS = 16
SCALE = 2048.0          # reward/value scale so targets are O(10)


class AfterstateValueNet(nn.Module):
    """CNN mapping an exponent board ``[N,4,4]`` -> scalar value ``V`` ``[N]``."""

    def __init__(self, num_channels: int = NUM_CHANNELS, hidden: int = 256):
        super().__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(num_channels, 128, kernel_size=2)   # 4x4 -> 3x3
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2)            # 3x3 -> 2x2
        self.fc1 = nn.Linear(128 * 2 * 2, hidden)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, boards_exp: torch.Tensor) -> torch.Tensor:
        x = boards_exp.long().clamp_(0, self.num_channels - 1)
        x = F.one_hot(x, self.num_channels).permute(0, 3, 1, 2).float()  # [N,C,4,4]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.value_head(x).squeeze(-1)                      # [N]


def _augment(boards: torch.Tensor) -> torch.Tensor:
    """Apply one random dihedral symmetry to a batch (cheap invariance training)."""
    k = int(torch.randint(4, (1,)).item())
    b = torch.rot90(boards, k, dims=(1, 2))
    if bool(torch.randint(2, (1,)).item()):
        b = torch.flip(b, dims=(2,))
    return b.contiguous()


@torch.no_grad()
def best_action(net, engine, states, scale=SCALE):
    """Greedy afterstate choice for a batch of states ``[N,4,4]``.

    Returns ``(chosen_after [N,4,4], chosen_reward [N], any_valid [N])``.
    """
    afters, rewards, changed = engine.all_afterstates(states)     # [4,N,..],[4,N],[4,N]
    n = states.shape[0]
    v = net(afters.reshape(4 * n, 4, 4)).reshape(4, n)            # scaled values
    q = rewards.float() / scale + v
    q = q.masked_fill(~changed, float("-inf"))
    best = q.argmax(0)                                            # [N]
    ar = torch.arange(n, device=states.device)
    return afters[best, ar], rewards[best, ar], changed.any(0)


@torch.no_grad()
def evaluate(net, engine, n_games=2000, scale=SCALE):
    """Play ``n_games`` in parallel, greedily, to termination. -> stats dict."""
    net.eval()
    boards = engine.new_boards(n_games)
    alive = torch.ones(n_games, dtype=torch.bool, device=boards.device)
    max_tiles = torch.zeros(n_games, dtype=torch.long, device=boards.device)
    scores = torch.zeros(n_games, device=boards.device)
    for _ in range(100000):
        if not bool(alive.any()):
            break
        after, reward, valid = best_action(net, engine, boards, scale)
        moved = valid & alive
        scores += torch.where(moved, reward.float(), torch.zeros_like(reward.float()))
        boards = torch.where(moved.view(-1, 1, 1), after, boards)
        engine.spawn(boards, mask=moved)
        max_tiles = torch.maximum(max_tiles, boards.amax(dim=(1, 2)))
        alive = alive & valid & ~engine.done(boards)
    net.train()
    mt = (1 << max_tiles).cpu()
    reach = {t: float((mt >= t).float().mean()) for t in (512, 1024, 2048, 4096, 8192)}
    return {"reach": reach, "mean_score": float(scores.mean()),
            "best_tile": int(mt.max()), "mean_max": float(mt.float().mean())}


def train(games_target=200_000, n_envs=1024, lr=1e-3, lr_final=None, scale=SCALE,
          device="cuda", grad_clip=10.0, net=None, target_sync=250, huber=True,
          log_cb=None, eval_cb=None, eval_every=20_000):
    """Online afterstate-TD training over ``n_envs`` parallel GPU games.

    ``net`` is created if not supplied (pass one in to checkpoint it mid-run).
    ``lr_final`` linearly decays the learning rate from ``lr`` over training.

    Stability (online value-net TD is prone to divergence — the "deadly triad"):
      * **target network** — the bootstrap value ``V(next_afterstate)`` is taken
        from a lagged copy synced every ``target_sync`` gradient steps, decoupling
        the regression target from the fast-moving online net (Double-style: the
        online net still *selects* the next move, the target net *values* it).
      * **Huber loss** — robust to the occasional large TD error.
    ``log_cb`` / ``eval_cb`` are optional progress callbacks. Returns the net.
    """
    engine = TorchVectorEngine(device)
    if net is None:
        net = AfterstateValueNet().to(device)
    target_net = copy.deepcopy(net).to(device)
    target_net.eval()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    boards = engine.new_boards(n_envs)
    games_done = 0
    step = 0
    next_eval = eval_every
    recent_max = torch.zeros(n_envs, dtype=torch.long, device=device)

    while games_done < games_target:
        if lr_final is not None:
            frac = min(1.0, games_done / games_target)
            for g in opt.param_groups:
                g["lr"] = lr + (lr_final - lr) * frac
        after, reward, valid = best_action(net, engine, boards, scale)
        s_next = after.clone()
        engine.spawn(s_next, mask=valid)
        done = engine.done(s_next)

        with torch.no_grad():
            # online net selects the next move; target net values it (Double-style).
            next_after, next_reward, next_valid = best_action(net, engine, s_next, scale)
            v_next = target_net(next_after)
        terminal = done | ~next_valid | ~valid
        target = torch.where(terminal, torch.zeros_like(v_next),
                             next_reward.float() / scale + v_next)

        pred = net(_augment(after))
        mask = valid.float()
        if huber:
            loss = (F.smooth_l1_loss(pred, target, reduction="none") * mask).sum() \
                / mask.sum().clamp(min=1.0)
        else:
            loss = ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()
        step += 1
        if step % target_sync == 0:
            target_net.load_state_dict(net.state_dict())

        # advance; track max tile; restart finished games.
        boards = s_next
        recent_max = torch.maximum(recent_max, boards.amax(dim=(1, 2)))
        finished = terminal
        n_fin = int(finished.sum())
        if n_fin:
            games_done += n_fin
            fin_max = (1 << recent_max[finished]).float()
            boards[finished] = engine.new_boards(n_fin)
            recent_max[finished] = 0
            if log_cb:
                log_cb(games_done, float(loss.item()), fin_max)
            if games_done >= next_eval:
                next_eval += eval_every
                if eval_cb:
                    eval_cb(games_done, evaluate(net, engine))
    return net
