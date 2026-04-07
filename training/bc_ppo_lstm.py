"""Behavioral cloning (CNN + LSTM policy) then recurrent PPO on BomberEnv.

Uses shared encoding and demo collection from bomber_shared.py, ICEC-style
rewards from reward_02.py. Supports --parallel_envs N for vectorized rollouts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import os
import random
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from engine import BomberEnv

try:
    from .bomber_shared import (
        AGENT_LOOKUP,
        NUM_ACTIONS,
        _make_agent,
        collect_demonstrations,
        encode_obs,
    )
    from .reward_02 import EpisodeRewardState, compute_reward_icec
    from .utils import plot_loss, plot_moving_average, plot_rewards
except ImportError:
    from bomber_shared import (
        AGENT_LOOKUP,
        NUM_ACTIONS,
        _make_agent,
        collect_demonstrations,
        encode_obs,
    )
    from reward_02 import EpisodeRewardState, compute_reward_icec
    from utils import plot_loss, plot_moving_average, plot_rewards

BC_CLASS_WEIGHTS = torch.tensor([0.3, 1.0, 1.0, 1.0, 1.0, 2.0], dtype=torch.float32)


class MapEncoder(nn.Module):
    """Light CNN: stem + 2 residual stages + global average pool (feat_dim = 2 * base)."""

    def __init__(self, map_shape):
        super().__init__()
        c, _, _ = map_shape

        class _SE(nn.Module):
            def __init__(self, channels: int, reduction: int = 16):
                super().__init__()
                hidden = max(channels // reduction, 4)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
                self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

            def forward(self, x):
                s = self.pool(x)
                s = F.relu(self.fc1(s), inplace=True)
                s = torch.sigmoid(self.fc2(s))
                return x * s

        class _ResBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.se = _SE(out_ch)
                self.act = nn.ReLU(inplace=True)
                self.proj = None
                if stride != 1 or in_ch != out_ch:
                    self.proj = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_ch),
                    )

            def forward(self, x):
                identity = x
                out = self.act(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = self.se(out)
                if self.proj is not None:
                    identity = self.proj(identity)
                out = out + identity
                return self.act(out)

        def _make_stage(in_ch: int, out_ch: int, blocks: int, stride: int):
            layers = [_ResBlock(in_ch, out_ch, stride=stride)]
            for _ in range(blocks - 1):
                layers.append(_ResBlock(out_ch, out_ch, stride=1))
            return nn.Sequential(*layers)

        base = 32
        blocks_per_stage = 1
        self.map_stem = nn.Sequential(
            nn.Conv2d(c, base, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        # Two stages after stem (no third downsample block): stem → s1 → s2 → GAP
        self.map_stage1 = _make_stage(base, base, blocks=blocks_per_stage, stride=1)
        self.map_stage2 = _make_stage(base, base * 2, blocks=blocks_per_stage, stride=2)
        self.map_pool = nn.AdaptiveAvgPool2d(1)
        self.map_dropout = nn.Dropout(p=0.1)
        self.feat_dim = base * 2

    def forward(self, map_x: torch.Tensor) -> torch.Tensor:
        x = self.map_stem(map_x)
        x = self.map_stage1(x)
        x = self.map_stage2(x)
        feat = self.map_pool(x).flatten(1)
        return self.map_dropout(feat)


class ActorCriticLSTM(nn.Module):
    def __init__(self, map_shape, aux_dim: int, num_actions: int,
                 lstm_hidden: int = 256, lstm_layers: int = 1):
        super().__init__()
        self.aux_dim = aux_dim
        self.num_actions = num_actions
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.encoder = MapEncoder(map_shape)
        lstm_in = self.encoder.feat_dim + aux_dim
        self.lstm = nn.LSTM(lstm_in, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.actor = nn.Linear(lstm_hidden, num_actions)
        self.critic = nn.Linear(lstm_hidden, 1)

    def init_hidden(self, batch_size: int, device):
        z = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        return z, z.clone()

    def forward_bc(self, map_x: torch.Tensor, aux_x: torch.Tensor):
        """I.i.d. BC: fresh LSTM state per batch."""
        b = map_x.shape[0]
        enc = self.encoder(map_x)
        x = torch.cat([enc, aux_x], dim=-1).unsqueeze(1)
        h0, c0 = self.init_hidden(b, map_x.device)
        out, _ = self.lstm(x, (h0, c0))
        h = out.squeeze(1)
        return self.actor(h), self.critic(h).squeeze(-1)

    def forward_step(
        self,
        map_x: torch.Tensor,
        aux_x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ):
        """Single timestep for N parallel envs: map/aux (N,...), hidden (h,c)."""
        enc = self.encoder(map_x)
        x = torch.cat([enc, aux_x], dim=-1).unsqueeze(1)
        out, new_hidden = self.lstm(x, hidden)
        h = out.squeeze(1)
        return self.actor(h), self.critic(h).squeeze(-1), new_hidden

    def forward_sequence(
        self,
        obs_map: torch.Tensor,
        obs_aux: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Rollout (T, N, ...); reset LSTM state per env when previous step ended.

        dones: (T, N) float/bool — done after transition at t.
        """
        t_max, n_env, c, h, w = obs_map.shape
        device = obs_map.device
        h_state, c_state = self.init_hidden(n_env, device)
        logits_list, values_list = [], []
        for t in range(t_max):
            if t > 0:
                d = dones[t - 1].float().view(1, n_env, 1)
                h_state = h_state * (1.0 - d)
                c_state = c_state * (1.0 - d)
            logits, val, (h_state, c_state) = self.forward_step(
                obs_map[t], obs_aux[t], (h_state, c_state)
            )
            logits_list.append(logits)
            values_list.append(val)
        return torch.stack(logits_list, dim=0), torch.stack(values_list, dim=0)


def pretrain_bc_lstm(
    model: ActorCriticLSTM,
    bc_data: dict,
    device: str,
    bc_epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_ratio: float = 0.1,
):
    n = len(bc_data["action"])
    idx = np.random.permutation(n)
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    weights = BC_CLASS_WEIGHTS.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6,
    )
    best_val = float("inf")
    best_sd = None
    history = []

    for epoch in range(bc_epochs):
        model.train()
        np.random.shuffle(train_idx)
        train_loss = train_n = 0
        for start in range(0, len(train_idx), batch_size):
            bi = train_idx[start:start + batch_size]
            m = torch.from_numpy(bc_data["map"][bi]).to(device)
            a = torch.from_numpy(bc_data["aux"][bi]).to(device)
            y = torch.from_numpy(bc_data["action"][bi]).to(device)
            logits, _ = model.forward_bc(m, a)
            loss = F.cross_entropy(logits, y, weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bi)
            train_n += len(bi)
        train_loss /= max(train_n, 1)
        history.append(train_loss)

        model.eval()
        val_loss = val_n = 0
        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size):
                bi = val_idx[start:start + batch_size]
                m = torch.from_numpy(bc_data["map"][bi]).to(device)
                a = torch.from_numpy(bc_data["aux"][bi]).to(device)
                y = torch.from_numpy(bc_data["action"][bi]).to(device)
                logits, _ = model.forward_bc(m, a)
                loss = F.cross_entropy(logits, y, weight=weights)
                val_loss += loss.item() * len(bi)
                val_n += len(bi)
        val_loss /= max(val_n, 1)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  BC epoch {epoch + 1}/{bc_epochs} train={train_loss:.4f} val={val_loss:.4f}")

    if best_sd is not None:
        model.load_state_dict(best_sd)
        model.to(device)
    model.train()
    print(f"  BC done — best val loss: {best_val:.4f}")
    return history


def save_checkpoint(path: str, model, optimizer, meta: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = dict(meta)
    meta.setdefault("num_actions", NUM_ACTIONS)
    payload = {
        "model_state_dict": model.state_dict(),
        "meta": meta,
        "agent_type": "bc_ppo_lstm",
        "num_actions": int(meta["num_actions"]),
    }
    if meta.get("input_spec") is not None:
        payload["input_shape"] = meta["input_spec"]
        payload["input_spec"] = meta["input_spec"]
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    print(f"Saved checkpoint {path}")


def load_checkpoint(path: str, model, device, optimizer=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Loaded checkpoint {path}")
    return ckpt.get("meta", {})


def is_bc_ppo_lstm_checkpoint(ckpt: dict) -> bool:
    """True for BC+PPO+LSTM saves (visualizer / eval)."""
    if ckpt.get("agent_type") == "bc_ppo_lstm":
        return True
    meta = ckpt.get("meta")
    if isinstance(meta, dict) and "lstm_hidden" in meta and "input_spec" in meta:
        return True
    return False


class BC_PPO_LSTM_Agent:
    """Greedy / ε-greedy policy wrapper for evaluation (carries LSTM state within an episode)."""

    def __init__(self, agent_id: int, checkpoint_path: str, device: str | None = None):
        self.agent_id = int(agent_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        meta = ckpt.get("meta", {})
        input_spec = meta.get("input_spec") or ckpt.get("input_shape")
        if input_spec is None:
            raise ValueError(f"Checkpoint {checkpoint_path!r} missing meta['input_spec'] or input_shape")
        map_shape = tuple(input_spec[0])
        aux_dim = int(input_spec[1])
        lstm_hidden = int(meta.get("lstm_hidden", 256))
        lstm_layers = int(meta.get("lstm_layers", 1))
        num_actions = int(meta.get("num_actions", ckpt.get("num_actions", NUM_ACTIONS)))
        self.map_shape = map_shape
        self.aux_dim = aux_dim
        self.num_actions = num_actions
        self.model = ActorCriticLSTM(
            map_shape, aux_dim, num_actions, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset_memory(self) -> None:
        """Call at each new episode so the LSTM does not carry over across games."""
        self._hidden = None

    def act(self, map_state, aux_state, epsilon: float = 0.0) -> int:
        if self._hidden is None:
            self._hidden = self.model.init_hidden(1, self.device)
        m = torch.from_numpy(map_state).float().unsqueeze(0).to(self.device)
        aux = torch.from_numpy(aux_state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _, self._hidden = self.model.forward_step(m, aux, self._hidden)
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        return int(logits.argmax(dim=-1).item())


def train_bc_ppo_lstm(
    user_id: int = 0,
    expert_type: str = "tactical",
    enemy_type: str = "simple",
    demo_episodes: int = 100,
    bc_epochs: int = 15,
    ppo_updates: int = 500,
    ppo_steps: int = 128,
    parallel_envs: int = 1,
    max_steps: int = 500,
    seed: int = 86,
    lstm_hidden: int = 256,
    lstm_layers: int = 1,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    ppo_epochs: int = 4,
    minibatch_size: int = 256,
    save_model: bool = True,
    load_checkpoint_path: str | None = None,
    skip_bc: bool = False,
    device: str | None = None,
    reward_log_episodes: int = 3,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    bc_data, _, input_spec = collect_demonstrations(
        expert_type=expert_type,
        opponent_type=enemy_type,
        num_episodes=demo_episodes,
        max_steps=max_steps,
        seed=seed,
        augment=True,
        store_dqfd_buffer=False,
        reward_fn=None,
    )
    if len(bc_data["action"]) == 0:
        print("No BC data — increase demo_episodes or weaken opponents.")
        return

    map_shape = tuple(input_spec[0])
    aux_dim = int(input_spec[1])
    model = ActorCriticLSTM(
        map_shape, aux_dim, NUM_ACTIONS,
        lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    tag = f"bcppo_{expert_type}_{enemy_type}_p{parallel_envs}_s{seed}"
    out_dir = f"ckpts/{tag}"

    bc_loss_history: list[float] = []
    if load_checkpoint_path:
        load_checkpoint(load_checkpoint_path, model, device, optimizer)
        model.train()
    elif not skip_bc:
        print("Phase 1: Behavioral cloning (LSTM policy, i.i.d. windows)")
        bc_loss_history = pretrain_bc_lstm(model, bc_data, device, bc_epochs=bc_epochs)
        if save_model:
            save_checkpoint(
                f"{out_dir}/after_bc.pth", model, None,
                {"input_spec": input_spec, "lstm_hidden": lstm_hidden, "lstm_layers": lstm_layers},
            )

    enemy_ids = [i for i in range(4) if i != user_id]
    enemy_agents = [_make_agent(enemy_type, agent_id=i) for i in enemy_ids]
    agent_ids = [user_id, *enemy_ids]

    n_env = max(1, int(parallel_envs))
    envs = [BomberEnv(max_steps=max_steps, seed=seed + i) for i in range(n_env)]

    rng = np.random.default_rng(seed + 90210)

    ep_reward_sum = [0.0] * n_env
    ep_comp_sum: list[defaultdict[str, float]] = [defaultdict(float) for _ in range(n_env)]
    ep_log_this = [False] * n_env
    reward_log_done = 0

    def reset_env_state(ei: int):
        obs = envs[ei].reset(seed=int(rng.integers(0, 2**31 - 1)))
        m, a = encode_obs(obs, agent_ids)
        ep = EpisodeRewardState()
        prev = None
        ep_reward_sum[ei] = 0.0
        ep_comp_sum[ei].clear()
        ep_log_this[ei] = reward_log_episodes > 0 and reward_log_done < reward_log_episodes
        return obs, m, a, ep, prev

    obs_l, map_l, aux_l, ep_l, prev_l = zip(*[reset_env_state(i) for i in range(n_env)])
    obs_l = list(obs_l)
    map_l = list(map_l)
    aux_l = list(aux_l)
    ep_l = list(ep_l)
    prev_l = list(prev_l)

    c, hh, ww = map_l[0].shape
    model.train()

    _epm = max(1, min(n_env, minibatch_size // max(1, ppo_steps)))
    print(
        f"Phase 2: PPO+LSTM  device={device}  parallel_envs={n_env}  "
        f"steps/rollout={ppo_steps}  recurrent_minibatch_envs={_epm} "
        f"(~{_epm * ppo_steps} transitions per opt step)"
    )

    ppo_loss_history: list[float] = []
    reward_history: list[float] = []
    rollout_return_history: list[float] = []

    pbar = tqdm(range(ppo_updates), desc="PPO+LSTM")
    for upd in pbar:
        # --- Rollout ---
        stor_m = np.zeros((ppo_steps, n_env, c, hh, ww), dtype=np.float32)
        stor_a = np.zeros((ppo_steps, n_env, aux_dim), dtype=np.float32)
        stor_act = np.zeros((ppo_steps, n_env), dtype=np.int64)
        stor_logp = np.zeros((ppo_steps, n_env), dtype=np.float32)
        stor_rew = np.zeros((ppo_steps, n_env), dtype=np.float32)
        stor_done = np.zeros((ppo_steps, n_env), dtype=np.float32)
        stor_val = np.zeros((ppo_steps, n_env), dtype=np.float32)

        h_s, c_s = model.init_hidden(n_env, device)

        for t in range(ppo_steps):
            m_t = torch.from_numpy(np.stack(map_l)).to(device)
            a_t = torch.from_numpy(np.stack(aux_l)).to(device)
            if t > 0:
                dprev = torch.from_numpy(stor_done[t - 1]).float().to(device).view(1, n_env, 1)
                h_s = h_s * (1.0 - dprev)
                c_s = c_s * (1.0 - dprev)

            logits, val, (h_s, c_s) = model.forward_step(m_t, a_t, (h_s, c_s))
            dist = Categorical(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions)

            stor_m[t] = np.stack(map_l)
            stor_a[t] = np.stack(aux_l)
            stor_act[t] = actions.cpu().numpy()
            stor_logp[t] = logp.detach().cpu().numpy()
            stor_val[t] = val.detach().cpu().numpy()

            for n in range(n_env):
                act_list = [None] * 4
                act_list[user_id] = int(stor_act[t, n])
                for e in enemy_agents:
                    act_list[e.agent_id] = e.act(obs_l[n])
                next_obs, term, trunc = envs[n].step(act_list)
                done = term or trunc
                if ep_log_this[n]:
                    comp_buf: dict[str, float] = {}
                    r, ep_l[n] = compute_reward_icec(
                        prev_l[n], next_obs, user_id, ep_l[n], out_components=comp_buf
                    )
                    ep_reward_sum[n] += r
                    for k, v in comp_buf.items():
                        if k not in ("dense_decay", "dense_applied"):
                            ep_comp_sum[n][k] += float(v)
                else:
                    r, ep_l[n] = compute_reward_icec(prev_l[n], next_obs, user_id, ep_l[n])
                stor_rew[t, n] = r
                stor_done[t, n] = float(done)
                if done:
                    if ep_log_this[n]:
                        print(
                            f"[reward components] finished_episode={reward_log_done + 1}/"
                            f"{reward_log_episodes} env={n} return={ep_reward_sum[n]:.4f}"
                        )
                        keys = sorted(ep_comp_sum[n].keys(), key=lambda x: (x == "total", x))
                        for k in keys:
                            print(f"    {k}: {ep_comp_sum[n][k]:.6f}")
                        reward_log_done += 1
                    obs_l[n], map_l[n], aux_l[n], ep_l[n], prev_l[n] = reset_env_state(n)
                else:
                    obs_l[n] = next_obs
                    map_l[n], aux_l[n] = encode_obs(next_obs, agent_ids)
                    prev_l[n] = next_obs

        reward_history.extend(stor_rew.reshape(-1).tolist())
        rollout_return_history.append(float(np.mean(stor_rew.sum(axis=0))))

        # Bootstrap value V(s_last) for GAE at T-1
        with torch.no_grad():
            m_last = torch.from_numpy(np.stack(map_l)).to(device)
            a_last = torch.from_numpy(np.stack(aux_l)).to(device)
            h_boot, c_boot = model.init_hidden(n_env, device)
            _, v_last, _ = model.forward_step(m_last, a_last, (h_boot, c_boot))

        rew_t = torch.from_numpy(stor_rew).to(device)
        done_t = torch.from_numpy(stor_done).to(device)
        val_t = torch.from_numpy(stor_val).to(device)
        old_logp = torch.from_numpy(stor_logp).to(device)

        next_v = v_last * (1.0 - done_t[-1])
        advantages = torch.zeros_like(rew_t)
        last_gae = torch.zeros(n_env, device=device)
        for t in reversed(range(ppo_steps)):
            if t == ppo_steps - 1:
                nv = next_v
            else:
                nv = val_t[t + 1]
            nonterm = 1.0 - done_t[t]
            delta = rew_t[t] + gamma * nv * nonterm - val_t[t]
            last_gae = delta + gamma * gae_lambda * nonterm * last_gae
            advantages[t] = last_gae
        returns = advantages + val_t

        adv_norm = advantages.clone()
        adv_norm = (adv_norm - adv_norm.mean()) / (adv_norm.std() + 1e-8)

        obs_map_t = torch.from_numpy(stor_m).to(device)
        obs_aux_t = torch.from_numpy(stor_a).to(device)
        act_t = torch.from_numpy(stor_act).long().to(device)

        # Recurrent minibatch: slice env columns so each minibatch runs full-length LSTM
        # (time × env) ≈ minibatch_size transitions per optimizer step.
        envs_per_mb = max(1, min(n_env, minibatch_size // max(1, ppo_steps)))

        pi_loss = v_loss = ent_scalar = torch.zeros((), device=device)
        for _ in range(ppo_epochs):
            env_order = np.random.permutation(n_env)
            n_mb = 0
            pi_acc = v_acc = ent_acc = 0.0
            for s in range(0, n_env, envs_per_mb):
                idx = env_order[s : s + envs_per_mb]
                idx_t = torch.tensor(idx, device=device, dtype=torch.long)

                mb_map = obs_map_t.index_select(1, idx_t)
                mb_aux = obs_aux_t.index_select(1, idx_t)
                mb_done = done_t.index_select(1, idx_t)
                mb_act = act_t.index_select(1, idx_t)
                mb_old_logp = old_logp.index_select(1, idx_t)
                mb_adv = adv_norm.index_select(1, idx_t)
                mb_ret = returns.index_select(1, idx_t)

                Tm, K = mb_map.shape[0], mb_map.shape[1]
                logits_full, values_full = model.forward_sequence(mb_map, mb_aux, mb_done)
                dist = Categorical(logits=logits_full.reshape(-1, NUM_ACTIONS))
                new_logp = dist.log_prob(mb_act.reshape(-1)).reshape(Tm, K)
                entropy = dist.entropy().reshape(Tm, K).mean()
                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_adv
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(values_full, mb_ret)
                loss = pi_loss + vf_coef * v_loss - ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                n_mb += 1
                pi_acc += pi_loss.item()
                v_acc += v_loss.item()
                ent_acc += entropy.item()

            pi_loss = torch.tensor(pi_acc / max(n_mb, 1), device=device)
            v_loss = torch.tensor(v_acc / max(n_mb, 1), device=device)
            ent_scalar = torch.tensor(ent_acc / max(n_mb, 1), device=device)

        total_obj = pi_loss + vf_coef * v_loss - ent_coef * ent_scalar
        ppo_loss_history.append(float(total_obj.item()))
        pbar.set_postfix(pi=f"{pi_loss.item():.3f}", v=f"{v_loss.item():.3f}", ent=f"{ent_scalar.item():.3f}")

    os.makedirs(out_dir, exist_ok=True)
    if bc_loss_history:
        plot_loss(bc_loss_history, save_path=f"{out_dir}/{tag}_bc_loss.png")
    plot_loss(ppo_loss_history, save_path=f"{out_dir}/{tag}_ppo_loss.png")
    plot_rewards(reward_history, save_path=f"{out_dir}/{tag}_rewards.png")
    plot_moving_average(
        rollout_return_history, window_size=10, save_path=f"{out_dir}/{tag}_moving_avg.png"
    )

    if save_model:
        save_checkpoint(
            f"{out_dir}/final.pth", model, optimizer,
            {
                "input_spec": input_spec,
                "lstm_hidden": lstm_hidden,
                "lstm_layers": lstm_layers,
                "parallel_envs": n_env,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC + Recurrent PPO (BomberEnv)")
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--user_id", type=int, default=0)
    parser.add_argument("--expert_type", type=str, default="tactical", choices=list(AGENT_LOOKUP.keys()))
    parser.add_argument("--enemy_type", type=str, default="simple", choices=list(AGENT_LOOKUP.keys()))
    parser.add_argument("--demo_episodes", type=int, default=100)
    parser.add_argument("--bc_epochs", type=int, default=15)
    parser.add_argument("--ppo_updates", type=int, default=500)
    parser.add_argument("--ppo_steps", type=int, default=128)
    parser.add_argument("--parallel_envs", type=int, default=1,
                        help="Number of parallel BomberEnv instances (1 = off)")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--lstm_hidden", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=256,
        help="Recurrent PPO: target transitions per minibatch; env columns per step = "
        "min(N, max(1, minibatch_size // ppo_steps)), each column runs full ppo_steps LSTM.",
    )
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--skip_bc", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--reward-log-episodes",
        type=int,
        default=3,
        help="Print per-component reward sums for the first N completed episodes (0 = off).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_bc_ppo_lstm(
        user_id=args.user_id,
        expert_type=args.expert_type,
        enemy_type=args.enemy_type,
        demo_episodes=args.demo_episodes,
        bc_epochs=args.bc_epochs,
        ppo_updates=args.ppo_updates,
        ppo_steps=args.ppo_steps,
        parallel_envs=args.parallel_envs,
        max_steps=args.max_steps,
        seed=args.seed,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        save_model=args.save_model,
        load_checkpoint_path=args.load_checkpoint,
        skip_bc=args.skip_bc,
        device=args.device,
        reward_log_episodes=args.reward_log_episodes,
    )
