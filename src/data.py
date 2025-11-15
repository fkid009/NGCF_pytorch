
import random
import numpy as np, pandas as pd
import scipy.sparse as sp
from typing import Tuple


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils import getDF
from src.path import RAW_DATA_DIR


class NGCFDataLoader:
    """
    Data loader for NGCF (Neural Graph Collaborative Filtering).

    Responsibilities:
    - Load raw interaction data (Amazon/Yelp)
    - Encode users/items into integer indices
    - Split into train/validation/test sets
    - Build interaction matrix R
    - Build NGCF adjacency matrices (adj_mat, norm_adj, mean_adj)
    - Provide BPR training samples (user, pos_item, neg_item)
    """

    def __init__(
        self,
        fname: str,
        source: str = "amazon",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        self.fname = fname
        self.source = source.lower()
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

        # -----------------------------
        # Validate split sizes
        # -----------------------------
        if not (0.0 < self.val_size < self.test_size < 1.0):
            raise ValueError(
                f"`val_size` must be < `test_size` < 1.0 (e.g., val=0.1, test=0.2). "
                f"Got val_size={self.val_size}, test_size={self.test_size}"
            )

        # -----------------------------
        # Load raw data and preprocess
        # -----------------------------
        self.raw_df = self._load_raw_data()
        (
            self.user_num,
            self.item_num,
            self.train,
            self.val,
            self.test,
        ) = self._preprocess()

        # -----------------------------
        # Build interaction matrix R and adjacency matrices
        # -----------------------------
        self.R = self._build_interaction_matrix()
        self.adj_mat, self.norm_adj, self.mean_adj = self._create_adj_mat()

        # Cache positive items per user (for fast BPR sampling)
        self._user_pos_items = (
            self.train.groupby("user")["item"].apply(list).to_dict()
        )
        self._all_users = np.array(list(self._user_pos_items.keys()))

    # ------------------------------------------------------------------
    # Raw data loader
    # ------------------------------------------------------------------
    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw dataset and return DataFrame with (user, item) columns."""
        if self.source == "amazon":
            fpath = RAW_DATA_DIR / f"{self.fname}.jsonl.gz"

            df = getDF(fpath)[["user_id", "asin"]].rename(
                columns={"user_id": "user", "asin": "item"}
            )
            return df

        if self.source == "yelp":
            raise ValueError("Yelp source is not implemented yet.")

        raise ValueError("Invalid `source`: choose 'amazon' or 'yelp'.")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _label_encode_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Label encode the specified column to consecutive integer IDs."""
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].values)
        return df

    def _preprocess(
        self,
    ) -> Tuple[int, int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocessing steps:
        1. Drop rows with NaN
        2. Label encode user/item columns
        3. Split into train/val/test sets
        """
        df = self.raw_df.copy()
        df = df.dropna(subset=["user", "item"])

        df = self._label_encode_column(df, "user")
        df = self._label_encode_column(df, "item")

        user_num = df["user"].nunique()
        item_num = df["item"].nunique()

        # Step 1: Split into train + (val+test)
        train, val_test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )

        # Step 2: Split val and test inside val_test proportionally
        val_ratio_in_val_test = self.val_size / self.test_size

        val, test = train_test_split(
            val_test,
            test_size=1.0 - val_ratio_in_val_test,
            random_state=self.random_state,
            shuffle=True,
        )

        return user_num, item_num, train, val, test

    # ------------------------------------------------------------------
    # Interaction matrix R
    # ------------------------------------------------------------------
    def _build_interaction_matrix(self) -> sp.dok_matrix:
        """Build R where R[u, i] = 1 for interactions in the training set."""
        R = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32)

        for row in self.train.itertuples(index=False):
            R[row.user, row.item] = 1.0

        return R

    # ------------------------------------------------------------------
    # NGCF adjacency matrix
    # ------------------------------------------------------------------
    @staticmethod
    def _normalized_adj_single(adj: sp.spmatrix) -> sp.coo_matrix:
        """
        Row-normalize adjacency matrix:
        D^{-1} A
        """
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        return d_mat_inv.dot(adj).tocoo()

    def _create_adj_mat(self):
        """
        Build adjacency matrix for NGCF:

            A = [[0,   R],
                 [R^T, 0]]

        Create:
        - mean_adj = D^{-1} A
        - norm_adj = D^{-1} (A + I)
        """
        n_users, n_items = self.R.shape

        adj_mat = sp.dok_matrix(
            (n_users + n_items, n_users + n_items), dtype=np.float32
        ).tolil()

        R = self.R.tolil()

        # Upper-right block: users → items
        adj_mat[:n_users, n_users:] = R

        # Lower-left block: items → users
        adj_mat[n_users:, :n_users] = R.T

        adj_mat = adj_mat.todok()

        norm_adj = self._normalized_adj_single(
            adj_mat + sp.eye(adj_mat.shape[0], dtype=np.float32)
        )
        mean_adj = self._normalized_adj_single(adj_mat)

        return adj_mat.tocsr(), norm_adj.tocsr(), mean_adj.tocsr()

    # ------------------------------------------------------------------
    # BPR Sampling
    # ------------------------------------------------------------------
    def _sample_negative_item(self, user: int) -> int:
        """Randomly sample a negative item that the user has not interacted with."""
        pos_items = set(self._user_pos_items[user])
        while True:
            neg_i = random.randint(0, self.item_num - 1)
            if neg_i not in pos_items:
                return neg_i

    def get_bpr_batch(self, batch_size: int):
        """
        Generate one batch for BPR training:
        - users: sampled with replacement
        - pos_items: random positive interaction for each user
        - neg_items: random negative item not interacted with
        """
        users = np.random.choice(self._all_users, size=batch_size, replace=True)

        pos_items = []
        neg_items = []

        for u in users:
            pos_i = random.choice(self._user_pos_items[u])
            neg_i = self._sample_negative_item(u)

            pos_items.append(pos_i)
            neg_items.append(neg_i)

        return (
            users.astype(np.int64),
            np.array(pos_items, dtype=np.int64),
            np.array(neg_items, dtype=np.int64),
        )
