#!/usr/bin/env python3
"""
test_jepa_representation.py — Unit tests for JEPA embedding quality

Five tests verifying the encoder learned meaningful representations:
  3. Temporal smoothness              (consecutive >= random similarity)
  4. Mask-ratio stability             (low vs high mask cos-sim high)
  5. Macro-feature correlation        (some emb dims correlate with VIX)
  6. Train/test effective-rank parity (PCA components match)
  8. Embedding-variance spread        (no dead dimensions)
"""

import sys
import random

import pytest
import torch
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, '.')
from src.models.encoder import Encoder
from individual_stocks.data_class_parquet import StockMarketJEPADataset
from individual_stocks.dataextraction import DataExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH    = "jepa-model/model_4_epoch_50.pt"
TRAIN_PARQUET = "individual_stocks/parquet_data/sector_etf_clean_trainingset.parquet"
TEST_PARQUET  = "individual_stocks/parquet_data/sector_etf_clean_testingset.parquet"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WINDOWS   = 200

ENC_DIM_IN      = 49
ENC_NUM_PATCHES = 20
ENC_KERNEL_SIZE = 49
ENC_EMBED_DIM   = 64
ENC_NHEAD       = 8
ENC_NUM_LAYERS  = 4

# fixtures ==================================================================

@pytest.fixture(scope="module", autouse=True)
def set_seeds():
    """Fix random seeds so mask sampling is deterministic across runs."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


@pytest.fixture(scope="module")
def encoder():
    """Load frozen encoder once per test session.

    Patches init_embed to avoid torch.from_numpy (broken numpy ABI).
    The checkpoint already supplies pos_embed so the patch is harmless.
    """
    _original_init_embed = Encoder.init_embed

    def _torch_init_embed(self):
        assert self.embed_dim % 2 == 0
        omega = 1.0 / (10000.0 ** (torch.arange(self.embed_dim // 2, dtype=torch.float32) / (self.embed_dim / 2)))
        pos = torch.arange(self.num_patches, dtype=torch.float32)
        out = torch.outer(pos, omega)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        self.pos_embed.data.copy_(emb.unsqueeze(0))

    Encoder.init_embed = _torch_init_embed
    try:
        enc = Encoder(
            dim_in=ENC_DIM_IN, num_patches=ENC_NUM_PATCHES,
            kernel_size=ENC_KERNEL_SIZE, embed_dim=ENC_EMBED_DIM,
            embed_bias=True, nhead=ENC_NHEAD, jepa=True,
            num_layers=ENC_NUM_LAYERS,
        )
    finally:
        Encoder.init_embed = _original_init_embed

    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    enc.load_state_dict(state)
    enc.to(DEVICE)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


@pytest.fixture(scope="module")
def train_ds():
    """Training dataset with deterministic mask sampling."""
    return StockMarketJEPADataset(
        mask_ratio=0.2, num_patches=ENC_NUM_PATCHES,
        vix_fairweather=20, parquet_path=TRAIN_PARQUET,
    )


@pytest.fixture(scope="module")
def test_ds():
    """Test dataset with deterministic mask sampling."""
    return StockMarketJEPADataset(
        mask_ratio=0.2, num_patches=ENC_NUM_PATCHES,
        vix_fairweather=20, parquet_path=TEST_PARQUET,
    )


@pytest.fixture(scope="module")
def train_emb(train_ds, encoder):
    """Pre-extracted pooled train embeddings: shape (N, 64)."""
    return _extract_embeddings(train_ds, encoder, MAX_WINDOWS)


@pytest.fixture(scope="module")
def test_emb(test_ds, encoder):
    """Pre-extracted pooled test embeddings: shape (N, 64)."""
    return _extract_embeddings(test_ds, encoder, MAX_WINDOWS)


# helpers ===================================================================

def _extract_embeddings(ds, encoder, max_windows):
    """
    Extract mean-pooled embeddings from non-overlapping windows.

    Example
    -------
    >>> emb = _extract_embeddings(dataset, encoder, 50)
    >>> emb.shape
    (50, 64)
    """
    windows = []
    n = min(len(ds), max_windows)
    with torch.no_grad():
        for i in range(n):
            window, _, _ = ds[i]
            windows.append(window.unsqueeze(0))
        x = torch.cat(windows, dim=0).to(DEVICE)
        emb = encoder(x)
        pooled = emb.mean(dim=1)
    return pooled.cpu().numpy()


def _cosine_similarity_matrix(emb):
    """Pairwise cosine similarity matrix.

    Example
    -------
    >>> e = np.array([[1,0],[0,1],[1,1]], dtype=float)
    >>> sim = _cosine_similarity_matrix(e)
    >>> sim.shape
    (3, 3)
    >>> sim[0,1]  # orthogonal
    0.0
    """
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb_norm @ emb_norm.T


# =========================== TESTS =========================================

class TestEmbeddingQuality:

    def test_03_temporal_smoothness(self, train_emb):
        """
        Consecutive time windows should be at least as similar as
        randomly paired windows because market data is temporally
        correlated.

        Toy example
        -----------
        Consecutive windows: cos-sim 0.85  vs  random pairs: cos-sim 0.40
        -> the encoder preserves temporal continuity.
        """
        sim_matrix = _cosine_similarity_matrix(train_emb)
        n = len(train_emb)

        consecutive = []
        for i in range(n - 1):
            consecutive.append(sim_matrix[i, i + 1])

        triu = np.triu(sim_matrix, k=2)
        random_pairs = triu[triu != 0]

        mu_consec = np.mean(consecutive)
        mu_rand = np.mean(random_pairs) if len(random_pairs) > 0 else 0.0

        print(f"\n   Consecutive cos-sim:  {mu_consec:.4f}")
        print(f"   Random-pair cos-sim:  {mu_rand:.4f}")
        assert mu_consec >= mu_rand, \
            f"Consecutive ({mu_consec:.3f}) lower than random ({mu_rand:.3f})"


    def test_04_mask_ratio_stability(self, encoder):
        """
        Embeddings should be nearly identical whether extracted with
        10% or 30% masking.  The encoder is robust to masking variation.

        Toy example
        -----------
        cos-sim between low-mask and high-mask embeddings = 0.94
        -> representation is stable under different mask intensities.
        """
        ds_low = StockMarketJEPADataset(
            mask_ratio=0.1, num_patches=ENC_NUM_PATCHES,
            vix_fairweather=20, parquet_path=TRAIN_PARQUET,
        )
        ds_high = StockMarketJEPADataset(
            mask_ratio=0.3, num_patches=ENC_NUM_PATCHES,
            vix_fairweather=20, parquet_path=TRAIN_PARQUET,
        )

        emb_low  = _extract_embeddings(ds_low, encoder, min(MAX_WINDOWS // 2, len(ds_low)))
        emb_high = _extract_embeddings(ds_high, encoder, min(MAX_WINDOWS // 2, len(ds_high)))

        n = min(len(emb_low), len(emb_high))
        sims = []
        for i in range(n):
            cos = np.dot(emb_low[i], emb_high[i]) / (
                np.linalg.norm(emb_low[i]) * np.linalg.norm(emb_high[i]) + 1e-8
            )
            sims.append(cos)
        avg_sim = np.mean(sims)

        print(f"\n   Low-vs-high mask cos-sim: {avg_sim:.4f}")
        assert avg_sim > 0.85, \
            f"Mask-ratio stability {avg_sim:.3f} below 0.85"


    def test_05_macro_correlation(self, train_emb):
        """
        Some embedding dimensions should correlate meaningfully
        with macro features (e.g., VIX).  This shows the encoder
        captures broad market signals.

        Toy example
        -----------
        Embed dim 12 has Pearson r=0.45 with VIX -> encoder tracks volatility.
        """
        ds = StockMarketJEPADataset(
            mask_ratio=0.2, num_patches=ENC_NUM_PATCHES,
            vix_fairweather=20, parquet_path=TRAIN_PARQUET,
        )

        max_abs_corr = 0.0
        n = min(MAX_WINDOWS, len(ds))
        for dim in range(train_emb.shape[1]):
            emb_ts = train_emb[:n, dim]
            vix_vals = []
            for i in range(n):
                window, _, _ = ds[i]
                last_patch_vix = window[-1, -1].item()  # VIX is last column
                vix_vals.append(last_patch_vix)
            if np.std(emb_ts) < 1e-8 or np.std(vix_vals) < 1e-8:
                continue
            corr = abs(np.corrcoef(emb_ts, vix_vals)[0, 1])
            if not np.isnan(corr):
                max_abs_corr = max(max_abs_corr, corr)

        print(f"\n   Max |r| (emb-dim vs VIX): {max_abs_corr:.4f}")
        assert max_abs_corr > 0.10, \
            f"Best dimension-VIX correlation {max_abs_corr:.3f} below 0.10"


    def test_06_train_test_consistency(self, train_emb, test_emb):
        """
        Train and test embeddings should have similar effective
        dimensionalities and a bounded mean shift, indicating the
        model generalizes without distribution collapse.

        Toy example
        -----------
        Train 90% PCA rank: 18, Test 90% PCA rank: 17  (close)
        Mean shift L2: 0.12                                  (small)
        """
        pca_train = PCA(n_components=min(50, train_emb.shape[1]))
        pca_train.fit(train_emb)
        train_90 = np.searchsorted(np.cumsum(pca_train.explained_variance_ratio_), 0.90) + 1

        pca_test = PCA(n_components=min(50, test_emb.shape[1]))
        pca_test.fit(test_emb)
        test_90 = np.searchsorted(np.cumsum(pca_test.explained_variance_ratio_), 0.90) + 1

        delta_rank = abs(train_90 - test_90)
        mean_shift = np.linalg.norm(train_emb.mean(axis=0) - test_emb.mean(axis=0))

        print(f"\n   Train 90% rank: {train_90}  |  Test 90% rank: {test_90}  (delta={delta_rank})")
        print(f"   Mean shift L2:  {mean_shift:.4f}")

        assert delta_rank <= 5, \
            f"Rank delta {delta_rank} too large"
        assert mean_shift < 1.0, \
            f"Mean shift {mean_shift:.3f} too large"


    def test_08_embedding_variance_spread(self, train_emb):
        """
        No embedding dimension should be dead (zero variance), and
        the coefficient of variation across dimensions should show
        reasonable spread.

        Toy example
        -----------
        Per-dim stds: min=0.05, max=0.40, mean=0.18  -> healthy spread
        """
        per_dim_std = np.std(train_emb, axis=0)

        min_std = per_dim_std.min()
        max_std = per_dim_std.max()
        mean_std = per_dim_std.mean()
        n_dead = int(np.sum(per_dim_std < 1e-6))

        cv = max_std / (mean_std + 1e-8)

        print(f"\n   Per-dim std — min: {min_std:.4f}  max: {max_std:.4f}  mean: {mean_std:.4f}")
        print(f"   Dead dims (<1e-6): {n_dead}")
        print(f"   CV (max/mean):     {cv:.3f}")

        assert n_dead == 0, \
            f"Found {n_dead} dead dimensions"
        assert cv > 1.5, \
            f"Variance too uniform (CV={cv:.2f}); expected > 1.5"
