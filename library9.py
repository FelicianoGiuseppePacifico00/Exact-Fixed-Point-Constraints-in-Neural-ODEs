import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
import json
import copy
import torch.nn.init as init
import torch.nn.functional as F



# ============================================================
# Attractor helpers (for Model_1_pseudoinverse and Model_1_QR)
# ============================================================

def _standardize_attractors_tensor(
    attractors: torch.Tensor,
    *,
    num_classes: int,
    n_visible: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns X_attr with shape [n_visible, num_classes] (columns are attractors).
    Accepts input as [C, n] or [n, C].
    """
    if not isinstance(attractors, torch.Tensor):
        raise TypeError("attractors must be a torch.Tensor")

    A = attractors.detach()
    if A.dim() != 2:
        raise ValueError(f"attractors must be 2D, got shape {tuple(A.shape)}")

    if A.shape == (num_classes, n_visible):
        X = A.t().contiguous()
    elif A.shape == (n_visible, num_classes):
        X = A.contiguous()
    else:
        raise ValueError(
            f"attractors must have shape [C,n]=({num_classes},{n_visible}) "
            f"or [n,C]=({n_visible},{num_classes}), got {tuple(A.shape)}"
        )

    X = X.to(device=device, dtype=dtype)
    return X

def _default_attractors_simple(
    *,
    num_classes: int,
    n_visible: int,
    dtype: torch.dtype,
    device: torch.device,
    radius: float = 1.0,   # kept for backward-compat signature, not used
    value: float = 0.8,    # NEW: ladder amplitude (what you want)
) -> torch.Tensor:
    """
    DEFAULT = "ladder" / block attractors (same spirit as the old Model_1):
      Each class c gets a block of coordinates set to `value`, others are 0.
    Output shape: [n_visible, num_classes].

    Requirement: n_visible // num_classes >= 1 (i.e., n_visible >= num_classes).
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be >= 1")
    if n_visible < num_classes:
        raise ValueError(
            f"Default ladder attractors require n_visible >= num_classes. "
            f"Got n_visible={n_visible}, num_classes={num_classes}. "
            f"Either pass `attractors=...` or increase n_visible."
        )

    X = torch.zeros(n_visible, num_classes, device=device, dtype=dtype)
    block = n_visible // num_classes  # >=1 guaranteed by check above

    for c in range(num_classes):
        start = c * block
        end = (c + 1) * block
        X[start:end, c] = value

    return X

def _assert_distinct_columns(X: torch.Tensor, tol: float = 1e-6):
    """
    X shape [n, C]. Checks columns are distinct with min pairwise distance > tol.
    """
    if X.dim() != 2:
        raise ValueError("X must be 2D [n,C].")
    C = X.shape[1]
    if C <= 1:
        return

    # Pairwise distances (C x C)
    D = torch.cdist(X.t(), X.t())
    D.fill_diagonal_(float("inf"))
    min_dist = float(D.min().detach().cpu().item())
    if not (min_dist > tol):
        raise ValueError(
            f"Attractors are not distinct enough: min pairwise distance={min_dist:.3e} <= tol={tol:.3e}."
        )


#===============================================================================================
#============================================= DATASETS  =======================================
#===============================================================================================

# =========================
# ====  Circle 2D  ========
# =========================

def make_circle_dataset(
    n_total: int = 6000,
    raggio_interno: float = 1.0,
    raggio_out_min: float = 1.2,
    r_out_max: float = 1.6,
    noise: float = 0.02,
    seed: int | None = 42,
    dtype=np.float32,
):
    """
    Returns X [N,2] float32 and y [N] int64.
    The split can be done deterministically via train_val_split(X,y, val_ratio) below.
    """
    rng = np.random.default_rng(seed)
    n_per = n_total // 2

    # Inside: uniform in disk
    u = rng.random(n_per)
    theta = 2 * np.pi * rng.random(n_per)
    r = raggio_interno * np.sqrt(u)
    x_in = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    # Outside: uniform in annulus
    u = rng.random(n_per)
    theta = 2 * np.pi * rng.random(n_per)
    r = np.sqrt((r_out_max**2 - raggio_out_min**2) * u + raggio_out_min**2)
    x_out = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    X = np.vstack([x_in, x_out]).astype(dtype, copy=False)
    y = np.concatenate([np.zeros(n_per, dtype=np.int64), np.ones(n_per, dtype=np.int64)])

    if noise and noise > 0:
        X += noise * rng.standard_normal(X.shape).astype(dtype, copy=False)

    # Shuffle once (reproducible with seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.25):
    """
    Deterministic split if X,y were pre-shuffled by a fixed seed.
    """
    N = len(X); n_val = int(val_ratio * N)
    return X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]

def _now_iso():
    return datetime.now().isoformat(timespec="seconds")

def save_dataset_npz(path: str | Path, X: np.ndarray, y: np.ndarray, meta: dict | None = None) -> str:
    """
    Save full dataset to a compressed .npz with metadata (as JSON string).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    meta.setdefault("saved_at", _now_iso())
    meta_json = json.dumps(meta)
    np.savez_compressed(path, X=X, y=y, meta_json=np.array(meta_json))
    return str(path)

def load_dataset_npz(path: str | Path):
    """
    Load full dataset .npz → (X, y, meta_dict)
    """
    with np.load(path, allow_pickle=False) as f:
        X = f["X"]
        y = f["y"]
        meta_json = str(f["meta_json"].item()) if "meta_json" in f else "{}"
    meta = json.loads(meta_json)
    return X, y, meta

def save_splits_npz(path: str | Path,
                    Xtr: np.ndarray, ytr: np.ndarray,
                    Xva: np.ndarray, yva: np.ndarray,
                    meta: dict | None = None) -> str:
    """
    Save train/val splits to a compressed .npz with metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    meta.setdefault("saved_at", _now_iso())
    meta_json = json.dumps(meta)
    np.savez_compressed(path, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, meta_json=np.array(meta_json))
    return str(path)

def load_splits_npz(path: str | Path):
    """
    Load train/val .npz → (Xtr, ytr, Xva, yva, meta_dict)
    """
    with np.load(path, allow_pickle=False) as f:
        Xtr = f["Xtr"]; ytr = f["ytr"]
        Xva = f["Xva"]; yva = f["yva"]
        meta_json = str(f["meta_json"].item()) if "meta_json" in f else "{}"
    meta = json.loads(meta_json)
    return Xtr, ytr, Xva, yva, meta

def get_or_make_circle_splits(
    path: str | Path,
    *,
    n_total: int = 20000,
    raggio_interno: float = 1.0,
    raggio_out_min: float = 1.2,
    r_out_max: float = 1.6,
    noise: float = 0.02,
    seed: int = 42,
    val_ratio: float = 0.25,
    overwrite: bool = False,
):
    """
    If 'path' exists and overwrite=False → load and return splits.
    Else → generate (deterministic by 'seed'), split, save, and return.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        return load_splits_npz(path)

    X, y = make_circle_dataset(
        n_total=n_total,
        raggio_interno=raggio_interno,
        raggio_out_min=raggio_out_min,
        r_out_max=r_out_max,
        noise=noise,
        seed=seed,
    )
    Xtr, ytr, Xva, yva = train_val_split(X, y, val_ratio=val_ratio)

    meta = {
        "dataset": "circle_2d",
        "n_total": n_total,
        "val_ratio": val_ratio,
        "raggio_interno": raggio_interno,
        "raggio_out_min": raggio_out_min,
        "r_out_max": r_out_max,
        "noise": noise,
        "seed": seed,
    }
    save_splits_npz(path, Xtr, ytr, Xva, yva, meta=meta)
    return Xtr, ytr, Xva, yva, meta









# =========================
# =====  Spirals 2D  ======
# =========================

def make_two_spirals_dataset(
    n_total: int = 6000,
    *,
    turns: float = 3.0,  # number of turns of the separating spiral pattern
    noise: float = 0.0,  # isotropic Gaussian noise in x,y
    minority_fraction: float = 0.50,  # if 0.5 classes are balanced
    minority_class: int = 1,          # which class is minority (0 or 1)
    seed: int | None = 42,
    dtype=np.float32,
):
    """Generate a *dense* 2D two-spirals dataset by uniform sampling + spiral decision boundary.

    We sample points uniformly in the unit disk, then label them with a spiral-shaped boundary:
        y = 1{ cos(theta - 2*pi*turns*r) < 0 }.

    This yields two *dense*, entangled spiral regions (no thin curves / empty interior).

    Returns
    -------
    X : [N,2] float32
    y : [N] int64
    """
    if not (0.0 < minority_fraction < 1.0):
        raise ValueError("minority_fraction must be in (0,1).")
    if minority_class not in (0, 1):
        raise ValueError("minority_class must be 0 or 1.")

    rng = np.random.default_rng(seed)

    # desired class counts
    n_min = int(round(n_total * minority_fraction))
    n_maj = n_total - n_min
    n0 = n_min if minority_class == 0 else n_maj
    n1 = n_total - n0

    # We generate a pool (typically balanced), then subsample to enforce imbalance exactly
    pool_n = int(max(n_total * 3, 2000))

    while True:
        # Uniform in unit disk: r = sqrt(u), theta = 2pi*v
        u = rng.random(pool_n)
        v = rng.random(pool_n)
        r = np.sqrt(u)
        theta = 2.0 * np.pi * v

        X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

        if noise and noise > 0:
            X = X + noise * rng.standard_normal(X.shape)

        # Spiral decision boundary: two entangled spiral regions
        phi = theta - (2.0 * np.pi * turns) * r
        y = (np.cos(phi) < 0.0).astype(np.int64)

        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]

        if len(idx0) >= n0 and len(idx1) >= n1:
            break

        # if extremely imbalanced requested, expand the pool and retry
        pool_n *= 2

    sel0 = rng.choice(idx0, size=n0, replace=False)
    sel1 = rng.choice(idx1, size=n1, replace=False)

    X = np.vstack([X[sel0], X[sel1]]).astype(dtype, copy=False)
    y = np.concatenate([np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)])

    # final shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def save_triplet_splits_npz(
    path: str | Path,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    meta: dict | None = None,
) -> str:
    """Save train/val/test splits to a compressed .npz with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    meta.setdefault("saved_at", _now_iso())
    meta_json = json.dumps(meta)
    np.savez_compressed(
        path,
        Xtr=Xtr, ytr=ytr,
        Xva=Xva, yva=yva,
        Xte=Xte, yte=yte,
        meta_json=np.array(meta_json),
    )
    return str(path)


def load_triplet_splits_npz(path: str | Path):
    """Load train/val/test .npz → (Xtr,ytr,Xva,yva,Xte,yte,meta_dict)"""
    with np.load(path, allow_pickle=False) as f:
        Xtr = f["Xtr"]; ytr = f["ytr"]
        Xva = f["Xva"]; yva = f["yva"]
        Xte = f["Xte"]; yte = f["yte"]
        meta_json = str(f["meta_json"].item()) if "meta_json" in f else "{}"
    meta = json.loads(meta_json)
    return Xtr, ytr, Xva, yva, Xte, yte, meta


def load_spirals_dataset(
    *,
    path: str | Path = "./data/spirals_2d_splits.npz",
    n_total: int = 12000,
    turns: float = 3.0,
    noise: float = 0.0,
    minority_fraction: float = 0.50,
    minority_class: int = 1,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    overwrite: bool = False,
    shuffle: bool = True,
    as_torch: bool = True,
):
    """Get (train, val, test) for the 2D two-spirals dataset with caching.

    If `path` exists and overwrite=False, loads splits from disk.
    Otherwise generates, shuffles, splits, saves, and returns.

    Returns
    -------
    train_ds, val_ds, test_ds, meta
        If as_torch=True: TensorDataset objects
        If as_torch=False: numpy arrays (Xtr,ytr,Xva,yva,Xte,yte,meta)

    Class imbalance is controlled by minority_fraction and minority_class.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        Xtr, ytr, Xva, yva, Xte, yte, meta = load_triplet_splits_npz(path)
    else:
        if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
            raise ValueError("Require val_ratio>=0, test_ratio>=0 and val_ratio+test_ratio < 1.")
        X, y = make_two_spirals_dataset(
            n_total=n_total,
            turns=turns,
            noise=noise,
            minority_fraction=minority_fraction,
            minority_class=minority_class,
            seed=seed,
        )

        if shuffle:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(X))
            X, y = X[idx], y[idx]

        N = len(X)
        n_test = int(round(test_ratio * N))
        n_val  = int(round(val_ratio * N))
        n_train = N - n_val - n_test
        if n_train <= 0:
            raise ValueError("Split sizes invalid: increase n_total or decrease val_ratio/test_ratio.")

        Xtr, ytr = X[:n_train], y[:n_train]
        Xva, yva = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        Xte, yte = X[n_train+n_val:], y[n_train+n_val:]

        meta = {
            "dataset": "spirals_2d",
            "n_total": int(n_total),
            "turns": float(turns),
            "noise": float(noise),
            "minority_fraction": float(minority_fraction),
            "minority_class": int(minority_class),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
            "seed": int(seed),
            "shuffle": bool(shuffle),
        }
        save_triplet_splits_npz(path, Xtr, ytr, Xva, yva, Xte, yte, meta=meta)

    if not as_torch:
        return Xtr, ytr, Xva, yva, Xte, yte, meta

    train_ds = TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(ytr).long())
    val_ds   = TensorDataset(torch.as_tensor(Xva), torch.as_tensor(yva).long())
    test_ds  = TensorDataset(torch.as_tensor(Xte), torch.as_tensor(yte).long())
    return train_ds, val_ds, test_ds, meta














#==================================================================================================
#===================================== Metodi di integrazione  ====================================
#==================================================================================================

class EulerIntegrator(nn.Module):
    def __init__(self, dt: float, steps: int):
        super().__init__()
        self.dt = float(dt)
        self.steps = int(steps)

    def forward(self, x0, vector_field_fn, return_trajectory: bool = False):
        x = x0
        if not return_trajectory:
            for _ in range(self.steps):
                x = x + self.dt * vector_field_fn(x)
            return x

        xs = [x]
        for _ in range(self.steps):
            x = x + self.dt * vector_field_fn(x)
            xs.append(x)
        return torch.stack(xs, dim=0)  # [K+1, B, n]

class RK4Integrator(nn.Module):
    """Classico Runge–Kutta 4th order, stessa interfaccia (forward)"""
    def __init__(self, dt: float, steps: int):
        super().__init__()
        self.dt = float(dt)
        self.steps = int(steps)

    def forward(self, x0, vector_field_fn):
        h = self.dt
        x = x0
        for _ in range(self.steps):
            k1 = vector_field_fn(x)
            k2 = vector_field_fn(x + 0.5 * h * k1)
            k3 = vector_field_fn(x + 0.5 * h * k2)
            k4 = vector_field_fn(x + h * k3)
            x = x + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x

# ---- Coupled integrators: work on (x, u) simultaneously -----
class CoupledEulerIntegrator(nn.Module):
    def __init__(self, dt: float, steps:int):
        super().__init__()
        self.dt = float(dt)
        self.steps = int(steps)

    def forward(self, x0: torch.Tensor, u0: torch.Tensor,  vector_field_fn):
        x, u = x0, u0
        h = self.dt
        for _ in range(self.steps):
            vector_field_x, vector_field_u = vector_field_fn(x, u) # two vector fields
            x = x + h * vector_field_x
            u = u + h * vector_field_u
        return x, u

class CoupledRK4Integrator(nn.Module):
    def __init__(self, dt: float, steps: int):
        super().__init__()
        self.dt = float(dt)
        self.steps = int(steps)

    def forward(self, x0: torch.Tensor, u0: torch.Tensor, vector_field_fn):
        x, u = x0, u0
        h = self.dt
        for _ in range(self.steps):
            k1x, k1u = vector_field_fn(x, u)
            k2x, k2u = vector_field_fn(x + 0.5*h*k1x, u + 0.5*h*k1u)
            k3x, k3u = vector_field_fn(x + 0.5*h*k2x, u + 0.5*h*k2u)
            k4x, k4u = vector_field_fn(x + h*k3x,   u + h*k3u)
            x = x + (h/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
            u = u + (h/6.0) * (k1u + 2*k2u + 2*k3u + k4u)
        return x, u



# ===============================================================================================
# ===================================== non linearities =========================================
# ===============================================================================================
class Squash(nn.Module):
    """
    f(x) = x^2 / (x^2 + c)
    Range: [0, 1)
    """
    def __init__(self, c: float = 1/8):
        super().__init__()
        # buffer -> si muove con .to(device) e va nel state_dict
        self.register_buffer('c', torch.tensor(float(c)))
        
        if self.c < 1/4.: 
            self.alphabet_values = [ 0 ,
                                    0.5 - torch.sqrt(1-4*self.c)/2. ,
                                    0.5 + torch.sqrt(1-4*self.c)/2.] 
        else: 
            self.alphabet_values = None

    def forward(self, x):
        xx = x * x
        return xx / (xx + self.c)
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        # f(x) = x^2 / (x^2 + c)  =>  f'(x) = 2x c / (x^2 + c)^2
        return (2.0 * x * self.c) / ((x * x + self.c) ** 2)

class SquashInverse(nn.Module):
    """
    Inversa (ramo non-negativo) della Squash:
      y = x^2 / (x^2 + c)  ->  x = sqrt(c * y / (1 - y))
    Definita per y in [0, 1).
    """
    def __init__(self, c: float = 1/8, eps: float = 1e-6):
        super().__init__()
        self.register_buffer('c', torch.tensor(float(c)))
        self.eps = float(eps)

    def forward(self, y):
        # numerically safe: clamp in [0, 1 - eps]
        y = torch.clamp(y, 0.0, 1.0 - self.eps)
        return torch.sqrt(self.c * y / (1.0 - y))


# ===============================================================================================
# ==========================HELPER TRAINING FUNCTIONS ===========================================
# ===============================================================================================

class _EarlyStoppingHelper:
    def __init__(self, metric="val_loss", mode="min", patience=10, min_delta=0.0, restore_best=True, verbose=False):
        assert metric in {"val_loss", "val_acc", "train_loss", "train_acc"}
        assert mode in {"min", "max"}
        self.metric = metric
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best = bool(restore_best)
        self.verbose = verbose

        self.best = None
        self.best_state = None
        self.bad_epochs = 0
        self.stopped_epoch = None

    def _better(self, score, best):
        if best is None:
            return True
        if self.mode == "min":
            return score < best - self.min_delta
        else:
            return score > best + self.min_delta

    def step(self, model, metrics_dict, epoch_idx):
        score = metrics_dict[self.metric]
        if self._better(score, self.best):
            self.best = score
            self.bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"→ New best {self.metric}: {score:.6f} at epoch {epoch_idx}")
            return False  # don't stop
        else:
            self.bad_epochs += 1
            if self.verbose:
                print(f"No improvement on {self.metric} ({self.bad_epochs}/{self.patience})")
            if self.bad_epochs >= self.patience:
                self.stopped_epoch = epoch_idx
                return True  # should stop
            return False

    def maybe_restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


# ===============================================================================================
# ========================================= MODELLI =============================================
# ===============================================================================================

class Model_1_pseudoinverse(nn.Module):
    r"""
        Model_1_pseudoinverse: Neural ODE classifier with *minimal* fixed-point planting using a pseudoinverse solve.

        Dynamics
        --------
        Same NODE form as Model_1:
            x_dot = -x + A1 f( A2 f2(x) + b2 ) + b1

        What is planted (minimal constraint)
        ------------------------------------
        We plant ONLY the fixed-point constraints on the chosen class prototypes {x̄_c}:
            F(x̄_c) = 0  for all c
        but we do NOT impose the stronger “orthogonal code” constraints of Model_1.

        We treat:
            A2 = W2   (fully free / trainable)
        and build A1 so that the constraints hold.

        Constraint system
        -----------------
        Define the feature vectors at the planted points:
            s_c = f( A2 f2(x̄_c) + b2 )  ∈ R^{m_hidden}
        Stack them into:
            S = [s_1 ... s_C] ∈ R^{m_hidden × C}
        and define target matrix:
            Y = [x̄_1 - b1 ... x̄_C - b1] ∈ R^{n_visible × C}

        Then fixed-point constraints are exactly:
            A1 S = Y

        Pseudoinverse construction
        --------------------------
        A particular solution is:
            A_part = Y S^+
        and the full solution set is:
            A1 = A_part + W1 (I - P),  where P = S S^+

        This class implements:
            S_pinv = pinv(S, rcond=self.pinv_rcond)
            P = S @ S_pinv
            A1 = (Y @ S_pinv) + W1 @ (I - P)

        Important implementation detail (performance)
        ---------------------------------------------
        Computing pinv(S) is expensive if done every ODE step. To avoid this:
        - forward(x0) computes A1 and A2 ONCE per forward pass
        - then integrates using a cached closure vf_cached(x)
        So pinv is computed once per batch update, not at every Euler/RK4 step.

        Training / inference flow (call graph)
        --------------------------------------
        Training (fit):
            out = self(xb)                           # forward -> builds A1 via pinv -> integrator
            target = x_attractors^T[yb]
            loss_task = mean(||out - target||^2)

        Optional stabilizer for rank(S)=C:
            loss += rank_stab_lambda * _rank_stabilizer_logdet()
        This discourages the feature columns s_c from collapsing (keeps S well-conditioned),
        which is important because the planting solve assumes rank(S)=C.

        Prediction/accuracy:
            pred = argmin_c ||out - x̄_c||^2  (nearest attractor)

        Inference:
            out = model(x0) returns x(T) and class by nearest prototype.

        Saving / loading
        ----------------
        save(...) stores:
            - state_dict
            - config (includes pinv_rcond)
            - attractors tensor
        load(...) reconstructs model + integrator from config and loads weights.

        Debug/analysis utilities
        ------------------------
        - vector_field(x): evaluates F(x) (rebuilds A1/A2; mainly for debugging)
        - _S_matrix(): builds S from current parameters
        - _rank_stabilizer_logdet(): differentiable conditioning regularizer on S
        - jacobian/jacobian_eig: local stability analysis
        """

    def __init__(self,
                 n_visible  : int,
                 m_hidden   : int,
                 num_classes: int, 

                 non_linearity = None,
                 hidden_non_linearity=None,

                 attractors: torch.Tensor | None = None,
                 attractor_value = None,

                 integrator: nn.Module = None,
                 pinv_rcond: float = 1e-6

                 ):
        super().__init__()
        """
            attractor_value is the value that the attractors have (the others are zero); 
            if you want the alphabet value just just pass it. 
        """
        
         
        self.num_classes = num_classes

        # network parameters 
        self.n_visible = n_visible
        self.m_hidden = m_hidden

      
        # -------------------------
        # Parameters (MUST exist before init.* calls)
        # -------------------------
        self.W1 = nn.Parameter(torch.empty(self.n_visible, self.m_hidden))
        self.b1 = nn.Parameter(torch.empty(self.n_visible))
        self.W2 = nn.Parameter(torch.empty(self.m_hidden, self.n_visible))
        self.b2 = nn.Parameter(torch.empty(self.m_hidden))

        # W2: drives features S, so make it well-conditioned
        init.orthogonal_(self.W2, gain=1.0)
        init.zeros_(self.b2)

        # W1: free part (nullspace term). Keep it smaller to avoid warping the field early.
        init.xavier_uniform_(self.W1, gain=0.3)   # smaller gain is a good default
        init.zeros_(self.b1)
        

        # nonlinearities
        self.f      = non_linearity            if non_linearity            is not None else Squash()
        # f2 non linearity
        self.f2     = hidden_non_linearity     if hidden_non_linearity     is not None else nn.Identity()
    
        # default ladder amplitude: use provided attractor_value if given,
        # otherwise try to use the last alphabet value of Squash, else fallback to 0.8
        if attractor_value is None:
            av = getattr(self.f, "alphabet_values", None)
            self.attractor_value = float(av[-1]) if (av is not None and len(av) > 0) else 0.8
        else:
            self.attractor_value = float(attractor_value)

        

        # integrator (pluggable; default = Euler)
        self.integrator = integrator if integrator is not None else EulerIntegrator(dt=0.03, steps=120)
        
        # =======  attractors only (fixed points)  ========

        self.build_x_attractors(attractors=attractors)  

        # pseudoinverse numerical threshold
        self.pinv_rcond = float(pinv_rcond)

        # in Model_1.__init__ AFTER building attractors/projections
        self.register_buffer('identity_n_visible', torch.eye(self.n_visible, dtype=self.W1.dtype))
        self.register_buffer('identity_m_hidden',  torch.eye(self.m_hidden,  dtype=self.W1.dtype))

        # Ensure S has full column rank at initialization by reinitializing W2,b2 if needed
        self._ensure_full_rank_S_at_init()

    def forward(self, x0: torch.Tensor):
        """
        Compute A1/A2 once per forward pass (batch), so we DO NOT compute pinv at every ODE step.
        """
        # Precompute matrices ONCE (A1 includes the pseudoinverse constraint)
        A2 = self._A2_matrix()   # [m, n] (here it's W2)
        A1 = self._A1_matrix()   # [n, m] (computes pinv once)

        # Cache tensors/modules for speed & clean closure
        b1 = self.b1
        b2 = self.b2
        f  = self.f
        f2 = self.f2

        def vf_cached(x: torch.Tensor) -> torch.Tensor:
            # vector_field = -x + A1 f(A2 f2(x) + b2) + b1
            return -x + f(f2(x) @ A2.T + b2) @ A1.T + b1

        return self.integrator(x0, vf_cached)

    
    def vector_field(self, x):
        # vector_field =  -x + A1 f(A2 f2(x) + b2) + b1; tipically we choose f2=Identity
        A1 = self._A1_matrix()
        A2 = self._A2_matrix()

        return -x + self.f( self.f2(x) @ (A2.T) + self.b2) @ A1.T + self.b1 
    


    def _A1_matrix(self) -> torch.Tensor:
        """
        Enforce fixed points via pseudoinverse:
          Want:  -x_attr_l + f(A2 f2(x_attr_l)+b2) A1^T + b1 = 0
          i.e.:  A1 S = (x_attr - b1 1^T)
    
        General solution (if rank(S)=C):
          A1 = Y S^+ + W1 (I - S S^+),  where Y = x_attr - b1 1^T
        """
        S = self._S_matrix()  # [m, C]
        Y = self.x_attractors - self.b1.unsqueeze(1)  # [n, C]
    
        # Moore–Penrose pseudoinverse
        S_pinv = torch.linalg.pinv(S, rcond=self.pinv_rcond)  # [C, m]
        P = S @ S_pinv                                        # [m, m]
    
        A1_base = Y @ S_pinv                                  # [n, m]
        return A1_base + self.W1 @ (self.identity_m_hidden - P)
    


    def _S_matrix(self) -> torch.Tensor:
        """
        S = [s_1,...,s_C] in R^{m x C},  s_l = f( A2 f2(x̄_l) + b2 ),
        where x̄_l are the planted attractors (columns of x_attractors).
        """
        A2 = self._A2_matrix()            # [m, n]
        Xc = self.x_attractors.t()        # [C, n]
        z  = self.f2(Xc) @ A2.t() + self.b2   # [C, m]
        s  = self.f(z)                    # [C, m]
        return s.t().contiguous()         # [m, C]

    def _rank_stabilizer_logdet(self, eps: float = 1e-6) -> torch.Tensor:
        """
        Conditioning stabilizer for S (m x C).
        Penalizes collapse of columns of S by maximizing det(S^T S).

        We use column-normalized S to avoid trivially increasing norms.
        Returns a scalar penalty (differentiable).
        """
        S = self._S_matrix()  # [m, C]
        # normalize columns
        col_norm = S.norm(dim=0, keepdim=True).clamp_min(1e-8)
        Sn = S / col_norm

        G = Sn.t() @ Sn  # [C, C]
        I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
        G = G + eps * I

        sign, logabsdet = torch.linalg.slogdet(G)
        # if numerical issues ever make sign <= 0, add a big penalty
        bad = (sign <= 0).to(G.dtype)
        return (-logabsdet) + bad * 1e6
    
    def _A2_matrix(self) -> torch.Tensor:
        return self.W2

    def _f_prime_hidden(self, z: torch.Tensor) -> torch.Tensor:
        """
        Elementwise derivative of f on hidden pre-activations z = A2 x + b2.
        Uses f.derivative(z) if available; otherwise falls back to autograd.
        Works for [m] or [B, m].
        """
        if hasattr(self.f, "derivative"):
            return self.f.derivative(z)

        # autograd fallback
        z_req = z.detach().requires_grad_(True)
        y = self.f(z_req)
        grad = torch.autograd.grad(
            outputs=y,
            inputs=z_req,
            grad_outputs=torch.ones_like(y),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        return grad

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        J(x) = d/dx[-x + A1 f(A2 x + b2) + b1] = -I + A1 diag(f'(A2 x + b2)) A2
        Returns:
        [n, n] if x is [n]
        [B, n, n] if x is [B, n]
        """
        A1 = self._A1_matrix()  # [n, m]
        A2 = self._A2_matrix()  # [m, n]
        I  = self.identity_n_visible  # [n, n]

        x_in = x if x.dim() == 2 else x.unsqueeze(0)     # [B, n]
        z    = x_in @ A2.T + self.b2                     # [B, m]
        fp   = self._f_prime_hidden(z)                   # [B, m]

        # Diag(fp) @ A2  == scale rows of A2 by fp
        A2_scaled = fp.unsqueeze(-1) * A2.unsqueeze(0)   # [B, m, n]
        J = (-I).unsqueeze(0) + torch.matmul(A1.unsqueeze(0), A2_scaled)  # [B, n, n]
        return J[0] if x.dim() == 1 else J

    def jacobian_eig(self, x: torch.Tensor, sort_by: str | None = "real", descending: bool = True) -> torch.Tensor:
        """
        Eigenvalues of J(x).
        sort_by: 'real' | 'abs' | None
        Returns:
        [n] complex if x is [n]
        [B, n] complex if x is [B, n]
        """
        J = self.jacobian(x)
        if J.dim() == 2:
            eig = torch.linalg.eigvals(J)
            if sort_by == "real":
                eig = eig[eig.real.argsort(descending=descending)]
            elif sort_by == "abs":
                eig = eig[eig.abs().argsort(descending=descending)]
            return eig
        else:
            try:
                eig = torch.linalg.eigvals(J)  # [B, n]
            except Exception:
                eig = torch.stack([torch.linalg.eigvals(Jb) for Jb in J], dim=0)
            if sort_by == "real":
                idx = eig.real.argsort(dim=-1, descending=descending); eig = torch.gather(eig, -1, idx)
            elif sort_by == "abs":
                idx = eig.abs().argsort(dim=-1, descending=descending); eig = torch.gather(eig, -1, idx)
            return eig



    #================================================================
    #========  building attractors and projection operators  ========
    #================================================================
    
        
    def build_x_attractors(self, attractors: torch.Tensor | None = None, distinct_tol: float = 1e-6):
        """
        - If `attractors` is provided: use them as x-space prototypes (any distinct C points).
        - If not provided: generate a simple default set of distinct points (circle/line).

        Stores:
        x_attractors: [n_visible, C]  (columns are attractors)
        """
        device = self.device
        dtype  = self.W1.dtype

        if attractors is None:
            X = _default_attractors_simple(
                num_classes=self.num_classes,
                n_visible=self.n_visible,
                dtype=dtype,
                device=device,
                value=float(self.attractor_value),
            )
        else:
            X = _standardize_attractors_tensor(
                attractors,
                num_classes=self.num_classes,
                n_visible=self.n_visible,
                dtype=dtype,
                device=device,
            )

        _assert_distinct_columns(X, tol=distinct_tol)

        # register buffer
        self.register_buffer("x_attractors", X)
        return X


    @torch.no_grad()
    def _reinit_Aux_bu(self):
        """
        Reinitialize A_ux and b_u (here: A2=W2 and b2) in a well-conditioned way.
        """
        init.orthogonal_(self.W2, gain=1.0)
        init.zeros_(self.b2)


    @torch.no_grad()
    def _ensure_full_rank_S_at_init(self, max_tries: int = 50, tol: float = 1e-4):
        """
        Ensures rank(S)=C at initialization by reinitializing W2,b2 until
        the Gram matrix S^T S has strictly positive smallest eigenvalue.

        tol is on singular values: we require sigma_min(S) > tol.
        Equivalent: lambda_min(S^T S) > tol^2.
        """
        if self.m_hidden < self.num_classes:
            raise ValueError(f"Need m_hidden >= num_classes to have rank(S)=C. Got m={self.m_hidden}, C={self.num_classes}.")

        for k in range(max_tries):
            S = self._S_matrix()                  # [m, C]
            G = (S.t() @ S).double()              # [C, C] (double for stability)
            lam_min = float(torch.linalg.eigvalsh(G).min().cpu().item())
            if lam_min > (tol * tol):
                return

            self._reinit_Aux_bu()

        raise RuntimeError(
            f"Could not initialize rank(S)=C after {max_tries} tries. "
            f"Try increasing m_hidden, changing activation, or increasing tol/max_tries."
        )

    
    #================================================================
    #===============  properties and helpers  =======================
    #================================================================
    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            print('no parameters, running on cpu')
            return torch.device('cpu')
        

    @torch.no_grad()
    def max_fixed_point_residual(self) -> float:
        A1 = self._A1_matrix()
        A2 = self._A2_matrix()
        x = self.x_attractors.t()  # [C, n]
        F = -x + self.f(self.f2(x) @ A2.t() + self.b2) @ A1.t() + self.b1
        return float(F.norm(dim=1).max().cpu().item())


    # ===== timing & UI helpers (inside Model_1) =====

    @staticmethod
    def _fmt_hms(sec: float) -> str:
        sec = max(0, int(sec))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _bar(p: float, length: int = 12) -> str:
        p = min(max(p, 0.0), 1.0)
        filled = int(round(p * length))
        return ("|" * filled) + ("-" * (length - filled))

    def _measure_train_batch_time(self, loader, device, optimizer, n_batches=3):
        """Warmup timing like your original: do a few *real* train steps."""
        import time
        self.train()
        it = iter(loader)
        times = []
        for _ in range(min(n_batches, len(loader))):
            try:
                xb, yb = next(it)
            except StopIteration:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            out = self(xb)
            target = self.x_attractors.t()[yb]
            loss = (out - target).pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return (sum(times) / len(times)) if times else None

    @torch.no_grad()
    def _measure_eval_batch_time(self, loader, device, n_batches=2):
        import time
        if loader is None or len(loader) == 0:
            return 0.0
        self.eval()
        it = iter(loader)
        times = []
        for _ in range(min(n_batches, len(loader))):
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            xb = xb.to(device, non_blocking=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = self(xb)  # forward only
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return (sum(times) / len(times)) if times else 0.0

    def _estimate_total_time_seconds(
        self, train_loader, device, epochs, optimizer, eval_loader=None,
        warmup_train_batches=3, warmup_eval_batches=2
    ):
        bt_train = self._measure_train_batch_time(train_loader, device, optimizer, n_batches=warmup_train_batches)
        if bt_train is None:
            return None
        bt_eval = self._measure_eval_batch_time(eval_loader, device, n_batches=warmup_eval_batches) if eval_loader is not None else 0.0
        per_epoch = len(train_loader) * bt_train + (len(eval_loader) * bt_eval if eval_loader is not None else 0.0)
        return per_epoch * epochs

    # ===== metrics & optimizer (inside Model_1) =====

    @torch.no_grad()
    def _closest_attractor_idx(self, outputs: torch.Tensor) -> torch.Tensor:
        """Argmin distance to class attractors."""
        A = self.x_attractors.t()  # [C, n_visible]
        d2 = (outputs.unsqueeze(1) - A.unsqueeze(0)).pow(2).sum(dim=-1)  # [B, C]
        return d2.argmin(dim=1)

    @torch.no_grad()
    def evaluate(self, loader, device: torch.device | None = None) -> dict:
        device = device or self.device
        self.eval()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()
            out = self(xb)
            targets = self.x_attractors.t()[yb]
            loss = (out - targets).pow(2).mean()
            pred = self._closest_attractor_idx(out)
            loss_sum += loss.item() * yb.numel()
            correct  += (pred == yb).sum().item()
            total    += yb.numel()
        return {"acc": (correct / max(total, 1)), "loss": (loss_sum / max(total, 1)), "n": total}

    def configure_optimizer(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        exclude_biases: tuple[str, ...] = (),
    ):
        """
        Build default Adam. To freeze biases, pass exclude_biases=('b1','b2') or any subset.
        """
        params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.endswith(ex) for ex in exclude_biases):
                continue
            params.append(p)
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)




        # ===== training loop (inside Model_1) =====
    
    def fit(
        self,
        train_loader,
        *,
        epochs: int = 30,
        device: torch.device | None = None,
        eval_loader=None,
        optimizer: torch.optim.Optimizer | None = None,
        grad_clip: float | None = None,
        verbose: bool = True,
        warmup_train_batches: int = 3,
        warmup_eval_batches: int = 2,
        
        # ---- rank/conditioning stabilizer for pseudoinverse planting ----
        rank_stab_lambda: float = 0.0,
        rank_stab_eps: float = 1e-6,
        rank_stab_every: int = 1,

        # ---- Early Stopping config ----
        early_stopping: dict | None = None,   # e.g. {"metric":"val_loss","mode":"min","patience":8,"min_delta":1e-4,"restore_best":True,"verbose":True}
    ):
        """
        Train with optional Early Stopping.
        Records train & val metrics:
        history["train_loss"], history["train_acc"], history["val_loss"], history["val_acc"]
        If early_stopping is provided, training halts when the monitored metric
        does not improve for 'patience' epochs; best weights are optionally restored.
        """
        import time

        device = device or self.device
        self.train()
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        optimizer = optimizer or self.configure_optimizer()

        # upfront total-time estimate (same helper you already use)
        est_total_sec0 = self._estimate_total_time_seconds(
            train_loader, device, epochs, optimizer,
            eval_loader=eval_loader,
            warmup_train_batches=warmup_train_batches,
            warmup_eval_batches=warmup_eval_batches,
        )
        if verbose:
            if est_total_sec0 is not None:
                print(f"ESTIMATED TOTAL TRAINING TIME ≈ {self._fmt_hms(est_total_sec0)}  (~{est_total_sec0/60:.1f} min)\n")
            else:
                print("Estimated total training time: N/A")

        # setup early stopping (helper already in your lib)
        es = None
        if early_stopping is not None:
            metric = early_stopping.get("metric", "val_loss")
            if metric.startswith("val_") and eval_loader is None:
                raise ValueError("early_stopping metric='val_*' requires eval_loader.")
            es = _EarlyStoppingHelper(
                metric=metric,
                mode=early_stopping.get("mode", "min"),
                patience=early_stopping.get("patience", 10),
                min_delta=early_stopping.get("min_delta", 0.0),
                restore_best=early_stopping.get("restore_best", True),
                verbose=early_stopping.get("verbose", False),
            )  # :contentReference[oaicite:2]{index=2}

        batches_per_epoch = len(train_loader)
        total_batches     = batches_per_epoch * epochs
        if total_batches == 0:
            if verbose:
                print("Empty train loader.")
            return history

        start_global = time.perf_counter()

        for epoch in range(1, epochs + 1):
            loss_sum, correct, count = 0.0, 0, 0

            for bidx, (xb, yb) in enumerate(train_loader, start=1):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()

                out = self(xb)
                target = self.x_attractors.t()[yb]
                
                loss_task = (out - target).pow(2).mean()
                loss = loss_task

                # rank/conditioning stabilizer (keeps rank(S)=C in practice)
                if rank_stab_lambda > 0.0 and (rank_stab_every <= 1 or (bidx % rank_stab_every == 0)):
                    stab = self._rank_stabilizer_logdet(eps=rank_stab_eps)
                    loss = loss + rank_stab_lambda * stab


                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()

                with torch.no_grad():
                    pred = self._closest_attractor_idx(out)
                    batch_correct = (pred == yb).sum().item()
                    batch_count   = yb.numel()

                loss_sum += loss.item() * batch_count
                correct  += batch_correct
                count    += batch_count

                # progress line with remaining time (whole-run estimate)
                if verbose:
                    epoch_p   = bidx / batches_per_epoch
                    avg_loss  = loss_sum / max(count, 1)
                    acc_pct   = 100.0 * (correct / max(count, 1))
                    if est_total_sec0 is not None:
                        elapsed   = time.perf_counter() - start_global
                        remaining = max(0.0, est_total_sec0 - elapsed)
                        remaining_txt = self._fmt_hms(remaining)
                    else:
                        remaining_txt = "N/A"
                    print(
                        f"\rEpoch {epoch:02d}/{epochs:02d} {self._bar(epoch_p)} {epoch_p*100:5.1f}%  "
                        f"loss {avg_loss:.4f}  acc {acc_pct:5.1f}%  "
                        f"|| Remaining time≈{remaining_txt}",
                        end="", flush=True
                    )
            if verbose:
                print()

            # aggregate epoch train metrics
            train_loss_epoch = loss_sum / max(count, 1)
            train_acc_epoch  = correct / max(count, 1)

            # validation (uses your existing evaluate -> returns {"acc","loss"})
            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader, device=device)  # :contentReference[oaicite:3]{index=3}
                val_acc  = eval_metrics["acc"]
                val_loss = eval_metrics["loss"]
                eval_msg = f"  val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%"
            else:
                val_loss = float('nan')
                val_acc  = float('nan')
                eval_msg = ""

            history["train_loss"].append(train_loss_epoch)
            history["train_acc"].append(train_acc_epoch)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose:
                print(f"Evaluation: train_loss={train_loss_epoch:.4f}  train_acc={train_acc_epoch*100:.2f}%{eval_msg}\n")

            # Early Stopping step
            if es is not None:
                monitor = {
                    "train_loss": train_loss_epoch,
                    "train_acc":  train_acc_epoch,
                    "val_loss":   val_loss,
                    "val_acc":    val_acc,
                }
                if es.step(self, monitor, epoch):
                    if verbose:
                        print(f"Early stopping at epoch {epoch} based on {es.metric}.")
                    break

        # restore best weights if requested
        if es is not None:
            es.maybe_restore(self)

        return history

    def save(self, path, optimizer: torch.optim.Optimizer | None = None, history: dict | None = None, extra: dict | None = None):
        from pathlib import Path
        from datetime import datetime as _dt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": (optimizer.state_dict() if optimizer is not None else None),
            "history": history,
            "config": {
                "class_name": self.__class__.__name__,
                "n_visible": self.n_visible,
                "m_hidden": self.m_hidden,
                "num_classes": self.num_classes,
                "integrator": self.integrator.__class__.__name__,
                "dt": getattr(self.integrator, "dt", None),
                "steps": getattr(self.integrator, "steps", None),
                "non_linearity": self.f.__class__.__name__,
                "pinv_rcond": float(self.pinv_rcond),
            },
            "x_attractors_tensor": self.x_attractors.detach().cpu(),
            "saved_at": _dt.now().isoformat(timespec="seconds"),
            "extra": extra,
        }
        torch.save(bundle, path)
        return str(path)

    @classmethod
    def load(cls, path, map_location="cpu", device: torch.device | None = None, strict: bool = True):
        bundle = torch.load(path, map_location=map_location)
        cfg = bundle["config"]

        # Recreate equivalent integrator
        if cfg.get("integrator", "") == "EulerIntegrator":
            integrator = EulerIntegrator(dt=cfg.get("dt", 0.03), steps=cfg.get("steps", 120))
        else:
            integrator = EulerIntegrator(dt=cfg.get("dt", 0.03), steps=cfg.get("steps", 120))

        model = cls(
            n_visible=cfg["n_visible"],
            m_hidden=cfg["m_hidden"],
            num_classes=cfg["num_classes"],
            non_linearity=Squash(),
            integrator=integrator,
            pinv_rcond=cfg.get("pinv_rcond", 1e-6),
        )
        if device is not None:
            model = model.to(device)

        model.load_state_dict(bundle["model_state_dict"], strict=strict)
        model.eval()
        return model, bundle

class Model_1_QR(nn.Module):
    r"""
        Model_1_QR: Neural ODE classifier with *minimal* fixed-point planting using QR decomposition (no pseudoinverse).

        Dynamics
        --------
        Same NODE:
            x_dot = -x + A1 f( A2 f2(x) + b2 ) + b1

        What is planted (minimal constraint)
        ------------------------------------
        We impose only:
            F(x̄_c) = 0  for all c
        with A2 free:
            A2 = W2

        As in the pseudoinverse class, define:
            s_c = f( A2 f2(x̄_c) + b2 ),  S = [s_1 ... s_C] ∈ R^{m × C}
            Y = [x̄_1 - b1 ... x̄_C - b1] ∈ R^{n × C}
        Constraint is:
            A1 S = Y

        QR construction (numerically stable; no explicit inverse)
        --------------------------------------------------------
        Compute thin QR of S:
            S = Q R
        where Q has orthonormal columns spanning span(S).
        If rank(S)=C (typical setting m ≥ C), R is C×C upper triangular.

        Then a particular constrained solution is obtained by solving:
            B R = Y     (triangular solve, no R^{-1} explicitly)
            A_part = B Q^T

        The full constrained solution set is:
            A1 = A_part + W1 (I - QQ^T)
        We implement the free nullspace term without forming an m×m matrix:
            W1(I - QQ^T) = W1 - (W1 Q) Q^T

        Stability option:
            qr_ridge adds a small diagonal to R before the triangular solve to improve conditioning
            (useful when S is close to rank-deficient).

        Important implementation detail (performance)
        ---------------------------------------------
        forward(x0) builds A1 (via QR + solve) and A2 once per forward pass,
        then integrates with a cached closure vf_cached(x).
        So QR/solve happens once per batch update, not at each integrator step.

        Training / inference flow (call graph)
        --------------------------------------
        Training (fit):
            out = self(xb)                           # forward -> builds A1 via QR -> integrator
            target = x_attractors^T[yb]
            loss_task = mean(||out - target||^2)

        Optional rank/conditioning stabilizer:
            loss += rank_stab_lambda * _rank_stabilizer_logdet()
        This helps keep S well-conditioned so QR stays stable and constraints remain enforceable.

        Prediction/accuracy:
            pred = argmin_c ||out - x̄_c||^2

        Inference:
            out = model(x0) returns x(T); class by nearest prototype.

        Saving / loading
        ----------------
        save(...) stores:
            - state_dict
            - config (includes qr_ridge)
            - attractors tensor
        load(...) reconstructs model + integrator from config and loads weights.

        Debug/analysis utilities
        ------------------------
        - _S_matrix(): builds S from current parameters
        - vector_field(x): evaluates F(x) (rebuilds A1/A2; debugging)
        - jacobian/jacobian_eig: local stability analysis
        """

    def __init__(self,
                 n_visible  : int,
                 m_hidden   : int,
                 num_classes: int, 

                 non_linearity = None,
                 hidden_non_linearity=None,

                 attractor_value = None,
                 attractors: torch.Tensor | None = None,

                 integrator: nn.Module = None,
                 qr_ridge: float = 0.0,   # small diagonal added to R for stability (recommended 1e-6..1e-4)
                ):
        super().__init__()
        """
            attractor_value is the value that the attractors have (the others are zero); 
            if you want the alphabet value just just pass it. 
        """
         
        self.num_classes = num_classes

        # network parameters 
        self.n_visible = n_visible
        self.m_hidden = m_hidden

        
        self.W1 = nn.Parameter(torch.empty(self.n_visible, self.m_hidden))
        self.b1 = nn.Parameter(torch.empty(self.n_visible))
        self.W2 = nn.Parameter(torch.empty(self.m_hidden, self.n_visible))
        self.b2 = nn.Parameter(torch.empty(self.m_hidden))

        # W2: drives features S, so make it well-conditioned
        init.orthogonal_(self.W2, gain=1.0)
        init.zeros_(self.b2)

        # W1: free part (nullspace term). Keep it smaller to avoid warping the field early.
        init.xavier_uniform_(self.W1, gain=0.3)   # smaller gain is a good default
        init.zeros_(self.b1)

        

        # nonlinearities
        self.f      = non_linearity            if non_linearity            is not None else nn.Sigmoid()
        # f2 non linearity
        self.f2     = hidden_non_linearity     if hidden_non_linearity     is not None else nn.Identity()
    

        # default ladder amplitude (old behavior): use provided attractor_value if given,
        # otherwise try to use the last alphabet value of Squash, else fallback to 0.8
        if attractor_value is None:
            av = getattr(self.f, "alphabet_values", None)
            self.attractor_value = float(av[-1]) if (av is not None and len(av) > 0) else 0.8
        else:
            self.attractor_value = float(attractor_value)


        # integrator (pluggable; default = Euler)
        self.integrator = integrator if integrator is not None else EulerIntegrator(dt=0.03, steps=120)
        
        # =======  attractors only (fixed points)  ========

        self.build_x_attractors(attractors=attractors)

        self.qr_ridge = float(qr_ridge)

        # in Model_1.__init__ AFTER building attractors/projections
        self.register_buffer('identity_n_visible', torch.eye(self.n_visible, dtype=self.W1.dtype))
        self.register_buffer('identity_m_hidden',  torch.eye(self.m_hidden,  dtype=self.W1.dtype))

        self._ensure_full_rank_S_at_init()

    def forward(self, x0: torch.Tensor):
        """
        Compute A1/A2 once per forward pass (batch), so we DO NOT compute pinv at every ODE step.
        """
        # Precompute matrices ONCE (A1 includes the pseudoinverse constraint)
        A2 = self._A2_matrix()   # [m, n] (here it's W2)
        A1 = self._A1_matrix()   # [n, m] (computes pinv once)

        # Cache tensors/modules for speed & clean closure
        b1 = self.b1
        b2 = self.b2
        f  = self.f
        f2 = self.f2

        def vf_cached(x: torch.Tensor) -> torch.Tensor:
            # vector_field = -x + A1 f(A2 f2(x) + b2) + b1
            return -x + f(f2(x) @ A2.T + b2) @ A1.T + b1

        return self.integrator(x0, vf_cached)

    def vector_field(self, x):
        # vector_field =  -x + A1 f(A2 f2(x) + b2) + b1; tipically we choose f2=Identity
        A1 = self._A1_matrix()
        A2 = self._A2_matrix()

        return -x + self.f( self.f2(x) @ (A2.T) + self.b2) @ A1.T + self.b1 
    
    def _A1_matrix(self) -> torch.Tensor:
        """
        Enforce fixed points via QR (no pinv):
        Want:  -x_attr_l + f(A2 f2(x_attr_l)+b2) A1^T + b1 = 0
        i.e.:  A1 S = Y,   where
                S = [s_1,...,s_C]  in R^{m x C},  s_l = f(A2 f2(x_attr_l) + b2)
                Y = x_attr - b1 1^T  in R^{n x C}

        If S has full column rank (typically m>=C and rank C):
        S = Q R  (thin QR), Q in R^{m x C}, R in R^{C x C} upper triangular
        Solve B R = Y  (triangular solve) and set A_part = B Q^T
        General solution: A1 = A_part + W1 (I - QQ^T)

        We implement W1(I-QQ^T) as: W1 - (W1 Q) Q^T (no big m×m matrix).
        """

        S = self._S_matrix()  # [m, C]
        Y = self.x_attractors - self.b1.unsqueeze(1)  # [n, C]

        # thin QR of S: S = Q R
        Q, R = torch.linalg.qr(S, mode="reduced")  # Q: [m, k], R: [k, C], k=min(m,C)

        # ---- Particular solution A_part such that A_part S = Y ----
        # Preferred case: R is square (happens when m >= C, so k=C) -> triangular solve.
        if R.shape[0] == R.shape[1]:
            k = R.shape[0]
            if self.qr_ridge > 0.0:
                R = R + self.qr_ridge * torch.eye(k, device=R.device, dtype=R.dtype)

            # Solve B R = Y  (B: [n, k])
            # Use transpose form to use solve_triangular:
            # (R^T) (B^T) = (Y^T)
            B_T = torch.linalg.solve_triangular(R.T, Y.T, upper=False)  # [k, n]
            B = B_T.T                                                   # [n, k]
            A_part = B @ Q.T                                            # [n, m]
        else:
            # Fallback: least-squares if m < C (or other edge cases)
            # Solve (S^T) A_part^T ≈ Y^T
            A_part_T = torch.linalg.lstsq(S.T, Y.T).solution            # [m, n]
            A_part = A_part_T.T                                         # [n, m]

        # ---- Free part that cannot affect constraints: W1(I - QQ^T) ----
        W1_perp = self.W1 - (self.W1 @ Q) @ Q.T                         # [n, m]

        return A_part + W1_perp

    def _S_matrix(self) -> torch.Tensor:
        """
        S = [s_1,...,s_C] in R^{m x C},  s_l = f( A2 f2(x̄_l) + b2 ),
        where x̄_l are the planted attractors (columns of x_attractors).
        """
        A2 = self._A2_matrix()            # [m, n]
        Xc = self.x_attractors.t()        # [C, n]
        z  = self.f2(Xc) @ A2.t() + self.b2   # [C, m]
        s  = self.f(z)                    # [C, m]
        return s.t().contiguous()         # [m, C]

    def _rank_stabilizer_logdet(self, eps: float = 1e-6) -> torch.Tensor:
        """
        Conditioning stabilizer for S (m x C).
        Penalizes collapse of columns of S by maximizing det(S^T S).

        We use column-normalized S to avoid trivially increasing norms.
        Returns a scalar penalty (differentiable).
        """
        S = self._S_matrix()  # [m, C]
        # normalize columns
        col_norm = S.norm(dim=0, keepdim=True).clamp_min(1e-8)
        Sn = S / col_norm

        G = Sn.t() @ Sn  # [C, C]
        I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
        G = G + eps * I

        sign, logabsdet = torch.linalg.slogdet(G)
        # if numerical issues ever make sign <= 0, add a big penalty
        bad = (sign <= 0).to(G.dtype)
        return (-logabsdet) + bad * 1e6
    
    def _A2_matrix(self) -> torch.Tensor:
        return self.W2

    def _f_prime_hidden(self, z: torch.Tensor) -> torch.Tensor:
        """
        Elementwise derivative of f on hidden pre-activations z = A2 x + b2.
        Uses f.derivative(z) if available; otherwise falls back to autograd.
        Works for [m] or [B, m].
        """
        if hasattr(self.f, "derivative"):
            return self.f.derivative(z)

        # autograd fallback
        z_req = z.detach().requires_grad_(True)
        y = self.f(z_req)
        grad = torch.autograd.grad(
            outputs=y,
            inputs=z_req,
            grad_outputs=torch.ones_like(y),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        return grad

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        J(x) = d/dx[-x + A1 f(A2 x + b2) + b1] = -I + A1 diag(f'(A2 x + b2)) A2
        Returns:
        [n, n] if x is [n]
        [B, n, n] if x is [B, n]
        """
        A1 = self._A1_matrix()  # [n, m]
        A2 = self._A2_matrix()  # [m, n]
        I  = self.identity_n_visible  # [n, n]

        x_in = x if x.dim() == 2 else x.unsqueeze(0)     # [B, n]
        z    = x_in @ A2.T + self.b2                     # [B, m]
        fp   = self._f_prime_hidden(z)                   # [B, m]

        # Diag(fp) @ A2  == scale rows of A2 by fp
        A2_scaled = fp.unsqueeze(-1) * A2.unsqueeze(0)   # [B, m, n]
        J = (-I).unsqueeze(0) + torch.matmul(A1.unsqueeze(0), A2_scaled)  # [B, n, n]
        return J[0] if x.dim() == 1 else J

    def jacobian_eig(self, x: torch.Tensor, sort_by: str | None = "real", descending: bool = True) -> torch.Tensor:
        """
        Eigenvalues of J(x).
        sort_by: 'real' | 'abs' | None
        Returns:
        [n] complex if x is [n]
        [B, n] complex if x is [B, n]
        """
        J = self.jacobian(x)
        if J.dim() == 2:
            eig = torch.linalg.eigvals(J)
            if sort_by == "real":
                eig = eig[eig.real.argsort(descending=descending)]
            elif sort_by == "abs":
                eig = eig[eig.abs().argsort(descending=descending)]
            return eig
        else:
            try:
                eig = torch.linalg.eigvals(J)  # [B, n]
            except Exception:
                eig = torch.stack([torch.linalg.eigvals(Jb) for Jb in J], dim=0)
            if sort_by == "real":
                idx = eig.real.argsort(dim=-1, descending=descending); eig = torch.gather(eig, -1, idx)
            elif sort_by == "abs":
                idx = eig.abs().argsort(dim=-1, descending=descending); eig = torch.gather(eig, -1, idx)
            return eig

    def stability_regularizer_sym(self, alpha: float = 0.0, hinge: str = "softplus") -> torch.Tensor:
        r"""
        Option A (symmetric-part) stability regularizer for planted fixed points.

        For each attractor x̄_c we compute the Jacobian J_c = DF(x̄_c) and its symmetric part:
            S_c = (J_c + J_c^T)/2.

        A sufficient condition for local exponential stability is:
            λ_max(S_c) ≤ -alpha.

        This returns a non-negative penalty:
            mean_c  φ( λ_max(S_c) + alpha )
        where φ is either 'softplus' (smooth) or 'relu2' (squared hinge).
        """
        X = self.x_attractors.t()            # [C, n]
        J = self.jacobian(X)                # [C, n, n]
        S = 0.5 * (J + J.transpose(-1, -2)) # symmetric part

        eigs = torch.linalg.eigvalsh(S)     # [C, n] ascending
        lam_max = eigs[..., -1]             # [C]

        v = lam_max + float(alpha)          # want v <= 0

        if hinge == "softplus":
            pen = F.softplus(v)
        elif hinge == "relu2":
            pen = F.relu(v).pow(2)
        else:
            raise ValueError("hinge must be 'softplus' or 'relu2'")

        return pen.mean()

    #================================================================
    #========  building attractors and projection operators  ========
    #================================================================
    
    def build_x_attractors(self, attractors: torch.Tensor | None = None, distinct_tol: float = 1e-6):
        """
        - If `attractors` is provided: use them as x-space prototypes (any distinct C points).
        - If not provided: generate a simple default set of distinct points (circle/line).

        Stores:
        x_attractors: [n_visible, C]  (columns are attractors)
        """
        device = self.device
        dtype  = self.W1.dtype

        if attractors is None:
            X = _default_attractors_simple(
                num_classes=self.num_classes,
                n_visible=self.n_visible,
                dtype=dtype,
                device=device,
                value=float(self.attractor_value),
            )
        else:
            X = _standardize_attractors_tensor(
                attractors,
                num_classes=self.num_classes,
                n_visible=self.n_visible,
                dtype=dtype,
                device=device,
            )

        _assert_distinct_columns(X, tol=distinct_tol)

        # register buffer
        self.register_buffer("x_attractors", X)
        return X
    
    @torch.no_grad()
    def _reinit_Aux_bu(self):
        """
        Reinitialize A_ux and b_u (here: A2=W2 and b2) in a well-conditioned way.
        """
        init.orthogonal_(self.W2, gain=1.0)
        init.zeros_(self.b2)

    @torch.no_grad()
    def _ensure_full_rank_S_at_init(self, max_tries: int = 50, tol: float = 1e-4):
        """
        Ensures rank(S)=C at initialization by reinitializing W2,b2 until
        the Gram matrix S^T S has strictly positive smallest eigenvalue.

        tol is on singular values: we require sigma_min(S) > tol.
        Equivalent: lambda_min(S^T S) > tol^2.
        """
        if self.m_hidden < self.num_classes:
            raise ValueError(f"Need m_hidden >= num_classes to have rank(S)=C. Got m={self.m_hidden}, C={self.num_classes}.")

        for k in range(max_tries):
            S = self._S_matrix()                  # [m, C]
            G = (S.t() @ S).double()              # [C, C] (double for stability)
            lam_min = float(torch.linalg.eigvalsh(G).min().cpu().item())
            if lam_min > (tol * tol):
                return

            self._reinit_Aux_bu()

        raise RuntimeError(
            f"Could not initialize rank(S)=C after {max_tries} tries. "
            f"Try increasing m_hidden, changing activation, or increasing tol/max_tries."
        )

    #================================================================
    #===============  properties and helpers  =======================
    #================================================================
    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            print('no parameters, running on cpu')
            return torch.device('cpu')
        
    @torch.no_grad()
    def max_fixed_point_residual(self) -> float:
        A1 = self._A1_matrix()
        A2 = self._A2_matrix()
        x = self.x_attractors.t()  # [C, n]
        F = -x + self.f(self.f2(x) @ A2.t() + self.b2) @ A1.t() + self.b1
        return float(F.norm(dim=1).max().cpu().item())
    
    # ================================================================
    # ================== timing & UI helpers =========================
    # ================================================================
    @staticmethod
    def _fmt_hms(sec: float) -> str:
        sec = max(0, int(sec))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _bar(p: float, length: int = 12) -> str:
        p = min(max(p, 0.0), 1.0)
        filled = int(round(p * length))
        return ("|" * filled) + ("-" * (length - filled))

    def _measure_train_batch_time(self, loader, device, optimizer, n_batches=3):
        """Warmup timing like your original: do a few *real* train steps."""
        import time
        self.train()
        it = iter(loader)
        times = []
        for _ in range(min(n_batches, len(loader))):
            try:
                xb, yb = next(it)
            except StopIteration:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            out = self(xb)
            target = self.x_attractors.t()[yb]
            loss = (out - target).pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return (sum(times) / len(times)) if times else None

    @torch.no_grad()
    def _measure_eval_batch_time(self, loader, device, n_batches=2):
        import time
        if loader is None or len(loader) == 0:
            return 0.0
        self.eval()
        it = iter(loader)
        times = []
        for _ in range(min(n_batches, len(loader))):
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            xb = xb.to(device, non_blocking=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = self(xb)  # forward only
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return (sum(times) / len(times)) if times else 0.0

    def _estimate_total_time_seconds(
        self, train_loader, device, epochs, optimizer, eval_loader=None,
        warmup_train_batches=3, warmup_eval_batches=2
    ):
        bt_train = self._measure_train_batch_time(train_loader, device, optimizer, n_batches=warmup_train_batches)
        if bt_train is None:
            return None
        bt_eval = self._measure_eval_batch_time(eval_loader, device, n_batches=warmup_eval_batches) if eval_loader is not None else 0.0
        per_epoch = len(train_loader) * bt_train + (len(eval_loader) * bt_eval if eval_loader is not None else 0.0)
        return per_epoch * epochs

    # ================================================================
    # ======================= metrics & optimizer  ===================
    # ================================================================
    @torch.no_grad()
    def _closest_attractor_idx(self, outputs: torch.Tensor) -> torch.Tensor:
        """Argmin distance to class attractors."""
        A = self.x_attractors.t()  # [C, n_visible]
        d2 = (outputs.unsqueeze(1) - A.unsqueeze(0)).pow(2).sum(dim=-1)  # [B, C]
        return d2.argmin(dim=1)

    @torch.no_grad()
    def evaluate(self, loader, device: torch.device | None = None) -> dict:
        device = device or self.device
        self.eval()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()
            out = self(xb)
            targets = self.x_attractors.t()[yb]
            loss = (out - targets).pow(2).mean()
            pred = self._closest_attractor_idx(out)
            loss_sum += loss.item() * yb.numel()
            correct  += (pred == yb).sum().item()
            total    += yb.numel()
        return {"acc": (correct / max(total, 1)), "loss": (loss_sum / max(total, 1)), "n": total}

    def configure_optimizer(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        exclude_biases: tuple[str, ...] = (),
    ):
        """
        Build default Adam. To freeze biases, pass exclude_biases=('b1','b2') or any subset.
        """
        params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.endswith(ex) for ex in exclude_biases):
                continue
            params.append(p)
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)




        # ===== training loop (inside Model_1) =====
    
    def fit(
        self,
        train_loader,
        *,
        epochs: int = 30,
        device: torch.device | None = None,
        eval_loader=None,
        optimizer: torch.optim.Optimizer | None = None,
        grad_clip: float | None = None,
        verbose: bool = True,
        warmup_train_batches: int = 3,
        warmup_eval_batches: int = 2,
        
        # ---- rank/conditioning stabilizer for pseudoinverse planting ----
        rank_stab_lambda: float = 0.0,
        rank_stab_eps: float = 1e-6,
        rank_stab_every: int = 1,

        # ---- fixed-point stability regularizer (Option A: symmetric-part contraction) ----
        stab_lambda: float = 0.0,
        stab_alpha: float = 0.0,
        stab_hinge: str = "softplus",   # 'softplus' (smooth) or 'relu2' (squared hinge)
        stab_every: int = 1,

        # ---- Early Stopping config ----
        early_stopping: dict | None = None,   # e.g. {"metric":"val_loss","mode":"min","patience":8,"min_delta":1e-4,"restore_best":True,"verbose":True}
    ):
        """
        Train with optional Early Stopping.
        Records train & val metrics:
        history["train_loss"], history["train_acc"], history["val_loss"], history["val_acc"]
        If early_stopping is provided, training halts when the monitored metric
        does not improve for 'patience' epochs; best weights are optionally restored.
        """
        import time

        device = device or self.device
        self.train()
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        optimizer = optimizer or self.configure_optimizer()

        # upfront total-time estimate (same helper you already use)
        est_total_sec0 = self._estimate_total_time_seconds(
            train_loader, device, epochs, optimizer,
            eval_loader=eval_loader,
            warmup_train_batches=warmup_train_batches,
            warmup_eval_batches=warmup_eval_batches,
        )
        if verbose:
            if est_total_sec0 is not None:
                print(f"ESTIMATED TOTAL TRAINING TIME ≈ {self._fmt_hms(est_total_sec0)}  (~{est_total_sec0/60:.1f} min)\n")
            else:
                print("Estimated total training time: N/A")

        # setup early stopping (helper already in your lib)
        es = None
        if early_stopping is not None:
            metric = early_stopping.get("metric", "val_loss")
            if metric.startswith("val_") and eval_loader is None:
                raise ValueError("early_stopping metric='val_*' requires eval_loader.")
            es = _EarlyStoppingHelper(
                metric=metric,
                mode=early_stopping.get("mode", "min"),
                patience=early_stopping.get("patience", 10),
                min_delta=early_stopping.get("min_delta", 0.0),
                restore_best=early_stopping.get("restore_best", True),
                verbose=early_stopping.get("verbose", False),
            )  # :contentReference[oaicite:2]{index=2}

        batches_per_epoch = len(train_loader)
        total_batches     = batches_per_epoch * epochs
        if total_batches == 0:
            if verbose:
                print("Empty train loader.")
            return history

        start_global = time.perf_counter()

        for epoch in range(1, epochs + 1):
            loss_sum, correct, count = 0.0, 0, 0

            for bidx, (xb, yb) in enumerate(train_loader, start=1):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()

                out = self(xb)
                target = self.x_attractors.t()[yb]
                
                loss_task = (out - target).pow(2).mean()
                loss = loss_task

                # rank/conditioning stabilizer (keeps rank(S)=C in practice)
                if rank_stab_lambda > 0.0 and (rank_stab_every <= 1 or (bidx % rank_stab_every == 0)):
                    stab = self._rank_stabilizer_logdet(eps=rank_stab_eps)
                    loss = loss + rank_stab_lambda * stab

                # fixed-point stability regularizer (Option A)
                if stab_lambda > 0.0 and (stab_every <= 1 or (bidx % stab_every == 0)):
                    stab_loss = self.stability_regularizer_sym(alpha=stab_alpha, hinge=stab_hinge)
                    loss = loss + stab_lambda * stab_loss


                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()

                with torch.no_grad():
                    pred = self._closest_attractor_idx(out)
                    batch_correct = (pred == yb).sum().item()
                    batch_count   = yb.numel()

                loss_sum += loss.item() * batch_count
                correct  += batch_correct
                count    += batch_count

                # progress line with remaining time (whole-run estimate)
                if verbose:
                    epoch_p   = bidx / batches_per_epoch
                    avg_loss  = loss_sum / max(count, 1)
                    acc_pct   = 100.0 * (correct / max(count, 1))
                    if est_total_sec0 is not None:
                        elapsed   = time.perf_counter() - start_global
                        remaining = max(0.0, est_total_sec0 - elapsed)
                        remaining_txt = self._fmt_hms(remaining)
                    else:
                        remaining_txt = "N/A"
                    print(
                        f"\rEpoch {epoch:02d}/{epochs:02d} {self._bar(epoch_p)} {epoch_p*100:5.1f}%  "
                        f"loss {avg_loss:.4f}  acc {acc_pct:5.1f}%  "
                        f"|| Remaining time≈{remaining_txt}",
                        end="", flush=True
                    )
            if verbose:
                print()

            # aggregate epoch train metrics
            train_loss_epoch = loss_sum / max(count, 1)
            train_acc_epoch  = correct / max(count, 1)

            # validation (uses your existing evaluate -> returns {"acc","loss"})
            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader, device=device)  # :contentReference[oaicite:3]{index=3}
                val_acc  = eval_metrics["acc"]
                val_loss = eval_metrics["loss"]
                eval_msg = f"  val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%"
            else:
                val_loss = float('nan')
                val_acc  = float('nan')
                eval_msg = ""

            history["train_loss"].append(train_loss_epoch)
            history["train_acc"].append(train_acc_epoch)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose:
                print(f"Evaluation: train_loss={train_loss_epoch:.4f}  train_acc={train_acc_epoch*100:.2f}%{eval_msg}\n")

            # Early Stopping step
            if es is not None:
                monitor = {
                    "train_loss": train_loss_epoch,
                    "train_acc":  train_acc_epoch,
                    "val_loss":   val_loss,
                    "val_acc":    val_acc,
                }
                if es.step(self, monitor, epoch):
                    if verbose:
                        print(f"Early stopping at epoch {epoch} based on {es.metric}.")
                    break

        # restore best weights if requested
        if es is not None:
            es.maybe_restore(self)

        return history

    def save(self, path, optimizer: torch.optim.Optimizer | None = None, history: dict | None = None, extra: dict | None = None):
        from pathlib import Path
        from datetime import datetime as _dt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": (optimizer.state_dict() if optimizer is not None else None),
            "history": history,
            "config": {
                "class_name": self.__class__.__name__,
                "n_visible": self.n_visible,
                "m_hidden": self.m_hidden,
                "num_classes": self.num_classes,
                "integrator": self.integrator.__class__.__name__,
                "dt": getattr(self.integrator, "dt", None),
                "steps": getattr(self.integrator, "steps", None),
                "non_linearity": self.f.__class__.__name__,
                "qr_ridge": float(self.qr_ridge),
            },
            "x_attractors_tensor": self.x_attractors.detach().cpu(),
            "saved_at": _dt.now().isoformat(timespec="seconds"),
            "extra": extra,
        }
        torch.save(bundle, path)
        return str(path)

    @classmethod
    def load(cls, path, map_location="cpu", device: torch.device | None = None, strict: bool = True):
        bundle = torch.load(path, map_location=map_location)
        cfg = bundle["config"]

        # Recreate equivalent integrator
        if cfg.get("integrator", "") == "EulerIntegrator":
            integrator = EulerIntegrator(dt=cfg.get("dt", 0.03), steps=cfg.get("steps", 120))
        else:
            integrator = EulerIntegrator(dt=cfg.get("dt", 0.03), steps=cfg.get("steps", 120))

        model = cls(
            n_visible=cfg["n_visible"],
            m_hidden=cfg["m_hidden"],
            num_classes=cfg["num_classes"],
            non_linearity=Squash(),
            integrator=integrator,
            qr_ridge=cfg.get("qr_ridge", 0.0),
        )
        if device is not None:
            model = model.to(device)

        model.load_state_dict(bundle["model_state_dict"], strict=strict)
        model.eval()
        return model, bundle










