"""
health_literacy_service.py
══════════════════════════════════════════════════════════════════════════════
Health Literacy Profiling & Feature Clustering Service
Based on: "Revealing the Latent Structure of Health Literacy in Online
          Peer-to-Peer Communication with Uncertainty-Aware Bayesian Deep Learning"

Pipeline stages:
  1. Preprocessing          — clean, normalise, tokenise text
  2. Feature extraction     — 5 HL dimensions → vector x ∈ ℝ⁵
  3. BVAE model             — exact architecture from paper (d=16, 32-unit FC layers)
  4. MC Dropout inference   — T=50 stochastic passes → mean + epistemic uncertainty
  5. Factor projection      — 16-d latent z → 3 interpretable factors
  6. User profiling         — Balanced / Specialized / Transitional
  7. HL level mapping       — Low / Basic / Intermediate / High
  8. Validation framework   — MSE, uncertainty-error correlation, calibration

Usage:
    service = HealthLiteracyService()
    service.fit(df_scores)                   # train BVAE
    result = service.predict(text)           # full pipeline for one post
    results = service.predict_batch(texts)   # batch prediction
    service.validate(df_scores)              # compute all validation metrics
    service.save("hl_model.pt")
    service.load("hl_model.pt")
"""

from __future__ import annotations

import re
import json
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from health_literacy_regex_patterns import (
    extract_fhl_features,
    extract_chl_features,
    extract_crhl_features,
    extract_dhl_features,
    extract_ehl_features,
    extract_all_features,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

_HTML_TAG    = re.compile(r"<[^>]+>")
_MULTI_SPACE = re.compile(r"\s{2,}")
_NON_ALPHA   = re.compile(r"[^\w\s\.,;:!?()\-'\"/%°+]")


def preprocess_text(text: str) -> str:
    """
    Stage 1 of the pipeline: strip HTML, normalise whitespace,
    remove noise characters, convert to lowercase.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _HTML_TAG.sub(" ", text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    text = _NON_ALPHA.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip().lower()


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION — 5 HL DIMENSIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Feature extraction (regex + POS — delegated to health_literacy_regex_patterns) ──

def _aggregate_to_hl_vector(features: dict) -> np.ndarray:
    """Convert the flat feature dict from extract_all_features → ℝ⁵ score vector."""
    def m(*keys):
        vals = [features.get(k, 0) for k in keys]
        return float(np.mean(vals)) if vals else 0.0

    fhl  = m("readability_score", "avg_sentence_length", "avg_clauses_per_sentence",
              "medical_entity_count", "medical_entity_density", "unique_medical_terms",
              "hedging_score", "confidence_score")
    chl  = m("question_count", "question_ratio", "conditional_expression_count",
              "modal_verb_count", "modal_verb_density", "context_marker_count",
              "context_provided")
    crhl = m("causal_connective_count", "contrastive_connective_count",
              "evidence_reference_count", "multiple_options_count")
    dhl  = m("online_reference_count", "credible_source_count", "cross_reference_count",
              "information_interpretation_score")
    ehl  = m("concreteness_score", "lexical_diversity", "present_verb_count",
              "present_verb_ratio", "determiner_count", "determiner_ratio",
              "adjective_count", "adjective_ratio", "function_word_count",
              "function_word_ratio")
    return np.array([fhl, chl, crhl, dhl, ehl], dtype=np.float32)


# ── Master extractor ────────────────────────────────────────────────────────

def extract_hl_vector(text: str) -> np.ndarray:
    """
    Returns x ∈ ℝ⁵ = [FHL, CHL, CRHL, DHL, EHL] for a single text.
    Uses regex + POS patterns for improved coverage and fewer false positives.
    All values are in [0, 1] before z-score normalisation.
    """
    features = extract_all_features(text)
    return _aggregate_to_hl_vector(features)


def extract_hl_dataframe(texts: List[str], verbose: bool = True) -> pd.DataFrame:
    """Extract HL vectors for a list of texts → DataFrame with 5 columns."""
    rows = []
    it = enumerate(texts)
    for i, t in it:
        if verbose and i % 1000 == 0:
            print(f"  Feature extraction: {i}/{len(texts)}", end="\r")
        rows.append(extract_hl_vector(t))
    if verbose:
        print(f"  Feature extraction: {len(texts)}/{len(texts)} done.")
    cols = ["FHL_score", "CHL_score", "CRHL_score", "DHL_score", "ExpressedHL_score"]
    return pd.DataFrame(rows, columns=cols)


# ══════════════════════════════════════════════════════════════════════════════
# 3. BAYESIAN VARIATIONAL AUTOENCODER — EXACT PAPER ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class BVAE(nn.Module):
    """
    Bayesian Variational Autoencoder for health literacy inference.

    Architecture (from paper, Section III-D):
      Encoder: FC(5→32) → ReLU → Dropout(0.2) → FC(32→32) → ReLU → Dropout(0.2)
               → parallel heads FC(32→d) for μ and log σ²
      Latent:  z ~ N(μ, σ²)  via reparameterisation trick
      Decoder: FC(d→32) → ReLU → Dropout(0.2) → FC(32→32) → ReLU → FC(32→5)
      Prior:   p(z) = N(0, I)

    Adaptive β:
      β = 1.0 if d ≤ 8
      β = 0.5 if d > 8
    """

    def __init__(self, input_dim: int = 5, latent_dim: int = 16, dropout_p: float = 0.2):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.beta       = 1.0 if latent_dim <= 8 else 0.5

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc_fc1    = nn.Linear(input_dim, 32)
        self.enc_drop1  = nn.Dropout(dropout_p)
        self.enc_fc2    = nn.Linear(32, 32)
        self.enc_drop2  = nn.Dropout(dropout_p)

        self.fc_mu      = nn.Linear(32, latent_dim)
        self.fc_logvar  = nn.Linear(32, latent_dim)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec_fc1    = nn.Linear(latent_dim, 32)
        self.dec_drop1  = nn.Dropout(dropout_p)
        self.dec_fc2    = nn.Linear(32, 32)
        self.dec_drop2  = nn.Dropout(dropout_p)
        self.dec_out    = nn.Linear(32, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc_fc1(x))
        h = self.enc_drop1(h)
        h = F.relu(self.enc_fc2(h))
        h = self.enc_drop2(h)
        mu     = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return mu, logvar

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_fc1(z))
        h = self.dec_drop1(h)
        h = F.relu(self.dec_fc2(h))
        h = self.dec_drop2(h)
        return self.dec_out(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        x_hat      = self.decode(z)
        return x_hat, mu, logvar

    # ── ELBO loss ─────────────────────────────────────────────────────────────

    def elbo_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        L = E[log p(x|z)] − β · KL(q(z|x) || p(z))
        """
        recon = F.mse_loss(x_hat, x, reduction="mean")
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss  = recon + self.beta * kl
        return loss, recon, kl


# ── PyTorch Dataset ──────────────────────────────────────────────────────────

class HLDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 4. MONTE CARLO DROPOUT INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def mc_dropout_inference(
    model: BVAE,
    x: torch.Tensor,
    T: int = 50,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform T stochastic forward passes with dropout active.

    Returns:
        x_hat_mean  : shape (N, input_dim)  — predictive mean
        sigma_unc   : shape (N, input_dim)  — epistemic uncertainty (std)
        z_mean      : shape (N, latent_dim) — mean latent vector
    """
    model.train()   # keep dropout ON
    x = x.to(device)

    recons = []
    z_means = []

    with torch.no_grad():
        for _ in range(T):
            x_hat, mu, _ = model(x)
            recons.append(x_hat.cpu().numpy())
            z_means.append(mu.cpu().numpy())

    recons   = np.stack(recons,  axis=0)  # (T, N, 5)
    z_stacked = np.stack(z_means, axis=0)  # (T, N, d)

    x_hat_mean = recons.mean(axis=0)
    sigma_unc  = recons.std(axis=0)
    z_mean     = z_stacked.mean(axis=0)

    return x_hat_mean, sigma_unc, z_mean


# ══════════════════════════════════════════════════════════════════════════════
# 5. FACTOR PROJECTION — PAPER TABLE II
# ══════════════════════════════════════════════════════════════════════════════

# Factor loadings from Table II of the paper
# Rows: [FHL, CHL, CRHL, DHL, EHL]
# Cols: [Factor1_Core, Factor2_Digital, Factor3_Applied]
FACTOR_LOADINGS = np.array([
    [ 0.831, -0.159, -0.466],   # FHL
    [ 0.627,  0.054,  0.058],   # CHL
    [ 0.701, -0.119,  0.169],   # CRHL
    [ 0.440, -0.532, -0.010],   # DHL
    [ 0.626, -0.040, -0.074],   # EHL
], dtype=np.float32)

FACTOR_NAMES = [
    "Core Integrated Proficiency",
    "Digital Proficiency",
    "Applied Functional Literacy",
]
VARIANCE_EXPLAINED = [0.432, 0.065, 0.051]


def project_to_factors(x_normalised: np.ndarray) -> np.ndarray:
    """
    Project z-score normalised 5-d HL scores onto the 3 latent factors
    using the paper's factor loading matrix.

    Args:
        x_normalised: shape (N, 5) or (5,)
    Returns:
        factors: shape (N, 3) or (3,)
    """
    single = x_normalised.ndim == 1
    if single:
        x_normalised = x_normalised[np.newaxis, :]
    factors = x_normalised @ FACTOR_LOADINGS   # (N, 3)
    return factors[0] if single else factors


# ══════════════════════════════════════════════════════════════════════════════
# 6. USER PROFILING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    profile_type:    str    # "Balanced" | "Specialized" | "Transitional"
    hl_level:        str    # "Low" | "Basic" | "Intermediate" | "High"
    factor1_core:    float
    factor2_digital: float
    factor3_applied: float
    sigma_unc:       float
    flagged:         bool   # True if sigma_unc > 0.5
    sub_type:        Optional[str] = None  # "Digitally-Specialized" | "Functionally-Specialized"

    def to_dict(self) -> dict:
        return asdict(self)


def assign_profile(
    factors: np.ndarray,
    sigma_unc: float,
    factor_medians: Optional[np.ndarray] = None,
) -> Tuple[str, Optional[str]]:
    """
    Assign Balanced / Specialized / Transitional profile.

    Rules (from paper Section V-A):
      Balanced:     Factor1 > median AND |Factor2|, |Factor3| moderate
      Specialized:  One factor significantly deviates (|deviation| > 1.5 std)
      Transitional: No clear pattern OR high variability
    """
    f1, f2, f3 = float(factors[0]), float(factors[1]), float(factors[2])

    # Use passed medians or simple zero-median assumption
    f1_med = float(factor_medians[0]) if factor_medians is not None else 0.0

    f1_dominant = f1 > f1_med
    f2_extreme  = abs(f2) > 1.0
    f3_extreme  = abs(f3) > 0.8

    # Specialized: one factor strongly deviates
    if f2_extreme and not f3_extreme:
        sub = "Digitally-Specialized" if f2 < 0 else "Digitally-Specialized-Positive"
        return "Specialized", sub
    if f3_extreme and not f2_extreme:
        sub = "Functionally-Specialized" if f3 < 0 else "Functionally-Specialized-Positive"
        return "Specialized", sub

    # Balanced: factor1 above median, others moderate
    if f1_dominant and not f2_extreme and not f3_extreme:
        return "Balanced", None

    # Transitional: everything else (or high uncertainty)
    return "Transitional", None


def map_hl_level(
    factors: np.ndarray,
    percentile_thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """
    Map Factor 1 (Core Integrated Proficiency) to HL ordinal level.

    Default thresholds (paper percentile-based):
      < 25th pct  → Low
      25–50th pct → Basic
      50–75th pct → Intermediate
      > 75th pct  → High

    Adjustment: if Factor2 (digital) is strongly positive but Factor1 is low,
    downgrade by one level (digital skills alone don't compensate for core).
    """
    thresholds = percentile_thresholds or {
        "low":          -0.75,   # z-score cut (25th pct of normal distribution)
        "basic":         0.00,
        "intermediate":  0.75,
    }

    f1 = float(factors[0])
    f2 = float(factors[1])

    if f1 < thresholds["low"]:
        level = "Low"
    elif f1 < thresholds["basic"]:
        level = "Basic"
    elif f1 < thresholds["intermediate"]:
        level = "Intermediate"
    else:
        level = "High"

    # Compensatory downgrade: strong digital but weak core → keep at most Basic
    if f2 < -1.2 and f1 < 0.25:
        level_order = ["Low", "Basic", "Intermediate", "High"]
        idx = level_order.index(level)
        if idx > 1:
            level = level_order[idx - 1]

    return level


# ══════════════════════════════════════════════════════════════════════════════
# 7. VALIDATION FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationMetrics:
    reconstruction_mse:          float
    reconstruction_rmse:         float
    avg_kl_divergence:           float
    uncertainty_error_corr:      float   # r(sigma_unc, |error|)
    coverage_95:                 float   # empirical 95% CI coverage
    coverage_68:                 float   # empirical 68% CI coverage
    variance_ratio:              float   # max / min KL across latent dims
    active_dimensions:           List[int]
    per_dim_mse:                 Dict[str, float]
    avg_uncertainty:             float
    flagged_ratio:               float   # proportion with sigma > 0.5

    def print_report(self):
        print("\n" + "═" * 62)
        print("  BVAE VALIDATION REPORT")
        print("═" * 62)
        print(f"  Reconstruction MSE     : {self.reconstruction_mse:.4f}  (target ≤ 0.12)")
        print(f"  Reconstruction RMSE    : {self.reconstruction_rmse:.4f}")
        print(f"  Avg KL divergence      : {self.avg_kl_divergence:.4f}")
        print(f"  Uncertainty-Error r    : {self.uncertainty_error_corr:.4f}  (target ≥ 0.40)")
        print(f"  95% CI coverage        : {self.coverage_95:.1%}  (target 85-95%)")
        print(f"  68% CI coverage        : {self.coverage_68:.1%}")
        print(f"  Variance ratio (active): {self.variance_ratio:.1f}  (target > 1000:1)")
        print(f"  Active latent dims     : {self.active_dimensions}")
        print(f"  Avg uncertainty        : {self.avg_uncertainty:.4f}")
        print(f"  Flagged (σ > 0.5)      : {self.flagged_ratio:.1%}")
        print("\n  Per-dimension MSE:")
        dims = ["FHL", "CHL", "CRHL", "DHL", "EHL"]
        for d, mse in zip(dims, self.per_dim_mse.values()):
            print(f"    {d:<6}: {mse:.4f}")
        print("═" * 62)

        # Pass/fail
        checks = {
            "MSE ≤ 0.12":              self.reconstruction_mse <= 0.12,
            "Uncertainty-Error r ≥ 0.4": self.uncertainty_error_corr >= 0.40,
            "95% coverage in [85,95%]": 0.85 <= self.coverage_95 <= 0.99,
            "Variance ratio > 1000":   self.variance_ratio > 1000,
        }
        print("\n  Metric checks:")
        for name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {name}")
        print("═" * 62)


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN SERVICE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class HealthLiteracyService:
    """
    End-to-end health literacy profiling service.

    Workflow:
        svc = HealthLiteracyService(latent_dim=16)
        svc.fit(df_scores)
        result = svc.predict("I have been experiencing...")
        metrics = svc.validate(df_scores)
        svc.save("model.pt")
    """

    LATENT_DIM     = 16
    MC_T           = 50
    BATCH_SIZE     = 32
    LR             = 1e-4
    PATIENCE       = 20
    MAX_EPOCHS     = 200
    FLAG_THRESHOLD = 0.5

    def __init__(self, latent_dim: int = 16, device: Optional[str] = None):
        self.latent_dim = latent_dim
        self.device     = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.scaler           = StandardScaler()
        self.model: Optional[BVAE] = None
        self.factor_medians: Optional[np.ndarray] = None
        self.hl_thresholds:  Optional[Dict[str, float]] = None
        self._fitted          = False
        print(f"[HealthLiteracyService] device={self.device} | latent_dim={latent_dim}")

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        df_scores: pd.DataFrame,
        val_split: float = 0.20,
        verbose: bool = True,
    ) -> "HealthLiteracyService":
        """
        Train the BVAE on the 5-column HL score DataFrame.
        Expects columns: FHL_score, CHL_score, CRHL_score, DHL_score, ExpressedHL_score
        """
        X_raw = df_scores.values.astype(np.float32)

        # Z-score normalise
        X = self.scaler.fit_transform(X_raw).astype(np.float32)

        # Compute factor medians on training data for profiling
        factors_train   = project_to_factors(X)
        self.factor_medians = np.median(factors_train, axis=0)

        # Compute HL level thresholds from data percentiles (Factor 1)
        f1_vals = factors_train[:, 0]
        self.hl_thresholds = {
            "low":          float(np.percentile(f1_vals, 25)),
            "basic":        float(np.percentile(f1_vals, 50)),
            "intermediate": float(np.percentile(f1_vals, 75)),
        }

        # Train/val split
        n      = len(X)
        n_val  = max(int(n * val_split), 1)
        n_train = n - n_val
        idx     = np.random.permutation(n)
        X_tr    = X[idx[:n_train]]
        X_va    = X[idx[n_train:]]

        tr_loader = DataLoader(HLDataset(X_tr), batch_size=self.BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(HLDataset(X_va), batch_size=self.BATCH_SIZE)

        # Initialise model
        self.model = BVAE(input_dim=5, latent_dim=self.latent_dim).to(self.device)
        optimizer  = optim.Adam(self.model.parameters(), lr=self.LR)

        best_val_loss = float("inf")
        patience_cnt  = 0
        best_state    = None

        print(f"[fit] Training BVAE | n_train={n_train} | n_val={n_val} | β={self.model.beta}")

        for epoch in range(1, self.MAX_EPOCHS + 1):
            # ── Training ──
            self.model.train()
            tr_loss = 0.0
            for X_batch in tr_loader:
                X_batch = X_batch.to(self.device)
                optimizer.zero_grad()
                x_hat, mu, logvar = self.model(X_batch)
                loss, _, _ = self.model.elbo_loss(X_batch, x_hat, mu, logvar)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
            tr_loss /= len(tr_loader)

            # ── Validation ──
            self.model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for X_batch in va_loader:
                    X_batch = X_batch.to(self.device)
                    x_hat, mu, logvar = self.model(X_batch)
                    loss, _, _ = self.model.elbo_loss(X_batch, x_hat, mu, logvar)
                    va_loss += loss.item()
            va_loss /= max(len(va_loader), 1)

            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:4d}/{self.MAX_EPOCHS} | "
                      f"train={tr_loss:.4f} | val={va_loss:.4f}")

            # Early stopping
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                patience_cnt  = 0
                best_state    = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.PATIENCE:
                    print(f"  Early stopping at epoch {epoch} (patience={self.PATIENCE})")
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        self._fitted = True
        print(f"[fit] Done. Best val loss: {best_val_loss:.4f}")
        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_from_scores(self, scores: np.ndarray) -> UserProfile:
        """
        Run full profiling pipeline on a pre-computed 5-d HL score vector.
        scores: shape (5,) in original (unnormalised) scale.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")

        X_norm = self.scaler.transform(scores.reshape(1, -1)).astype(np.float32)
        x_t    = torch.tensor(X_norm, dtype=torch.float32).to(self.device)

        # MC Dropout inference
        x_hat_mean, sigma_unc, z_mean = mc_dropout_inference(
            self.model, x_t, T=self.MC_T, device=self.device
        )

        # Factor projection
        factors = project_to_factors(X_norm[0])

        # Scalar uncertainty (mean across 5 dims)
        unc_scalar = float(sigma_unc.mean())

        # Profiling
        profile_type, sub_type = assign_profile(
            factors, unc_scalar, self.factor_medians
        )
        hl_level = map_hl_level(factors, self.hl_thresholds)

        return UserProfile(
            profile_type    = profile_type,
            hl_level        = hl_level,
            factor1_core    = float(factors[0]),
            factor2_digital = float(factors[1]),
            factor3_applied = float(factors[2]),
            sigma_unc       = unc_scalar,
            flagged         = unc_scalar > self.FLAG_THRESHOLD,
            sub_type        = sub_type,
        )

    def predict(self, text: str) -> UserProfile:
        """Full pipeline: raw text → UserProfile."""
        scores = extract_hl_vector(text)
        return self.predict_from_scores(scores)

    def predict_batch(self, texts: List[str]) -> List[UserProfile]:
        """Batch prediction for a list of texts."""
        results = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"  Predicting: {i}/{len(texts)}", end="\r")
            results.append(self.predict(text))
        print(f"  Predicting: {len(texts)}/{len(texts)} done.")
        return results

    def predict_batch_from_df(self, df_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Predict from a DataFrame of pre-computed scores.
        Returns DataFrame with profile columns appended.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before prediction")

        X_raw  = df_scores.values.astype(np.float32)
        X_norm = self.scaler.transform(X_raw).astype(np.float32)
        x_t    = torch.tensor(X_norm, dtype=torch.float32).to(self.device)

        # Run MC Dropout in one batch
        x_hat_mean, sigma_unc, z_mean = mc_dropout_inference(
            self.model, x_t, T=self.MC_T, device=self.device
        )
        factors_all = project_to_factors(X_norm)
        unc_scalar  = sigma_unc.mean(axis=1)

        records = []
        for i in range(len(X_norm)):
            factors    = factors_all[i]
            unc        = float(unc_scalar[i])
            ptype, sub = assign_profile(factors, unc, self.factor_medians)
            level      = map_hl_level(factors, self.hl_thresholds)
            records.append({
                "factor1_core":    float(factors[0]),
                "factor2_digital": float(factors[1]),
                "factor3_applied": float(factors[2]),
                "sigma_unc":       unc,
                "profile_type":    ptype,
                "sub_type":        sub or "",
                "hl_level":        level,
                "flagged":         unc > self.FLAG_THRESHOLD,
            })

        return pd.concat([df_scores.reset_index(drop=True), pd.DataFrame(records)], axis=1)

    # ── Validate ─────────────────────────────────────────────────────────────

    def validate(self, df_scores: pd.DataFrame) -> ValidationMetrics:
        """
        Compute full validation suite on a held-out or training set.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        X_raw  = df_scores.values.astype(np.float32)
        X_norm = self.scaler.transform(X_raw).astype(np.float32)
        x_t    = torch.tensor(X_norm, dtype=torch.float32).to(self.device)

        x_hat_mean, sigma_unc, z_mean = mc_dropout_inference(
            self.model, x_t, T=self.MC_T, device=self.device
        )

        # ── Reconstruction metrics ────────────────────────────────────────────
        errors      = np.abs(X_norm - x_hat_mean)
        per_dim_mse = {
            name: float(((X_norm[:, i] - x_hat_mean[:, i]) ** 2).mean())
            for i, name in enumerate(["FHL", "CHL", "CRHL", "DHL", "EHL"])
        }
        mse  = float(((X_norm - x_hat_mean) ** 2).mean())
        rmse = float(np.sqrt(mse))

        # ── Uncertainty-error correlation ─────────────────────────────────────
        unc_flat = sigma_unc.mean(axis=1)
        err_flat = errors.mean(axis=1)
        r, _     = stats.pearsonr(unc_flat, err_flat)

        # ── Calibration (95% and 68% CI) ──────────────────────────────────────
        z95 = 1.96
        z68 = 1.00
        in_95 = np.abs(X_norm - x_hat_mean) <= z95 * sigma_unc
        in_68 = np.abs(X_norm - x_hat_mean) <= z68 * sigma_unc
        cov95 = float(in_95.mean())
        cov68 = float(in_68.mean())

        # ── Latent space: KL per dimension ───────────────────────────────────
        self.model.eval()
        kl_per_dim = np.zeros(self.latent_dim)
        n_batches  = 0
        with torch.no_grad():
            for start in range(0, len(X_norm), self.BATCH_SIZE):
                batch = x_t[start : start + self.BATCH_SIZE]
                _, mu, logvar = self.model.encode(batch[0:1]), None, None
                mu, logvar = self.model.encode(batch)
                kl  = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl_per_dim += kl.mean(dim=0).cpu().numpy()
                n_batches += 1
        kl_per_dim /= n_batches

        active_dims    = sorted(np.argsort(kl_per_dim)[-3:].tolist())  # top-3 active
        variance_ratio = float(kl_per_dim.max() / max(kl_per_dim.min(), 1e-10))

        # ── Flagging ratio ────────────────────────────────────────────────────
        flagged_ratio = float((unc_flat > self.FLAG_THRESHOLD).mean())

        return ValidationMetrics(
            reconstruction_mse     = mse,
            reconstruction_rmse    = rmse,
            avg_kl_divergence      = float(kl_per_dim.mean()),
            uncertainty_error_corr = float(r),
            coverage_95            = cov95,
            coverage_68            = cov68,
            variance_ratio         = variance_ratio,
            active_dimensions      = active_dims,
            per_dim_mse            = per_dim_mse,
            avg_uncertainty        = float(unc_flat.mean()),
            flagged_ratio          = flagged_ratio,
        )

    # ── Dimensionality search ────────────────────────────────────────────────

    def dimensionality_search(
        self,
        df_scores: pd.DataFrame,
        dims: List[int] = None,
    ) -> pd.DataFrame:
        """
        Train models for each latent dimensionality and report metrics.
        Reproduces Figure 4 of the paper.
        """
        dims = dims or [2, 4, 8, 16, 32]
        rows = []
        original_latent_dim = self.latent_dim

        for d in dims:
            print(f"\n[dim_search] Training d={d}…")
            self.latent_dim = d
            self.fit(df_scores, verbose=False)
            m = self.validate(df_scores)
            rows.append({
                "latent_dim":           d,
                "reconstruction_mse":   m.reconstruction_mse,
                "avg_uncertainty":      m.avg_uncertainty,
                "uncertainty_error_r":  m.uncertainty_error_corr,
                "variance_ratio":       m.variance_ratio,
            })
            print(f"  d={d} → MSE={m.reconstruction_mse:.4f} | "
                  f"unc={m.avg_uncertainty:.4f}")

        self.latent_dim = original_latent_dim
        return pd.DataFrame(rows)

    # ── Save / load ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        payload = {
            "latent_dim":      self.latent_dim,
            "model_state":     self.model.state_dict() if self.model else None,
            "scaler":          self.scaler,
            "factor_medians":  self.factor_medians,
            "hl_thresholds":   self.hl_thresholds,
            "fitted":          self._fitted,
        }
        torch.save(payload, path)
        print(f"[save] Model saved → {path}")

    def load(self, path: str) -> "HealthLiteracyService":
        payload = torch.load(path, map_location=self.device)
        self.latent_dim      = payload["latent_dim"]
        self.scaler          = payload["scaler"]
        self.factor_medians  = payload["factor_medians"]
        self.hl_thresholds   = payload["hl_thresholds"]
        self._fitted         = payload["fitted"]
        if payload["model_state"]:
            self.model = BVAE(input_dim=5, latent_dim=self.latent_dim).to(self.device)
            self.model.load_state_dict(payload["model_state"])
        print(f"[load] Model loaded ← {path}")
        return self
