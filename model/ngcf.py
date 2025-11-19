import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(
        self,
        user_num: int,
        item_num: int,
        L: torch.Tensor,
        embed_dim: int,
        n_layer: int, 
        dropout: float,
        l2_reg: float,
        negative_slope: float = 0.2,
    ):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.node_num = self.user_num + self.item_num
        self.embed_dim = embed_dim
        self.n_layer = n_layer
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.negative_slope = negative_slope

        # normalized adjacency (sparse)
        self.L = L.coalesce()

        # node embeddings: users + items
        self.embedding = nn.Embedding(self.node_num, embed_dim)

        # layer-wise parameters
        self.W1 = nn.ParameterList()
        self.W2 = nn.ParameterList()

        for _ in range(self.n_layer):
            w1 = nn.Parameter(torch.empty(embed_dim, embed_dim))
            w2 = nn.Parameter(torch.empty(embed_dim, embed_dim))
            nn.init.xavier_uniform_(w1)
            nn.init.xavier_uniform_(w2)
            self.W1.append(w1)
            self.W2.append(w2)

        # init embedding weights
        nn.init.xavier_uniform_(self.embedding.weight)

    def propagate(self):
        """
        Multi-layer NGCF propagation.

        Returns
        -------
        E_user : (user_num, embed_dim * (n_layer + 1))
        E_item : (item_num, embed_dim * (n_layer + 1))
        """

        E0 = self.embedding.weight  # (node_num, embed_dim)
        E_list = [E0]

        E_prev = E0

        for layer in range(self.n_layer):
            # message passing: side information from neighbors
            side_E = torch.sparse.mm(self.L, E_prev)  # (node_num, embed_dim)

            # (L + I) E_prev = side_E + E_prev
            sum_E = side_E + E_prev

            # bi-interaction term: element-wise product
            bi_E = E_prev * side_E

            # linear transforms: (sum_E @ W1) + (bi_E @ W2)
            E_sum = sum_E @ self.W1[layer]   # (node_num, embed_dim)
            E_bi  = bi_E  @ self.W2[layer]   # (node_num, embed_dim)

            E_next_pre = E_sum + E_bi

            # LeakyReLU
            E_next = F.leaky_relu(E_next_pre, negative_slope=self.negative_slope)

            # dropout
            if self.dropout > 0.0 and self.training:
                E_next = F.dropout(E_next, p=self.dropout, training=self.training)

            E_prev = E_next
            E_list.append(E_prev)

        # concat [E0, E1, ..., E_L] along dim=1
        E_all = torch.cat(E_list, dim=1)  # (node_num, embed_dim * (n_layer + 1))

        # split to user/item
        E_user = E_all[: self.user_num]
        E_item = E_all[self.user_num :]

        return E_user, E_item

    def forward(self, user_idx, item_idx):
        E_user, E_item = self.propagate()
        return E_user[user_idx], E_item[item_idx]

    def predict(self, user_idx, item_idx):
        user_emb, item_emb = self.forward(user_idx, item_idx)
        return (user_emb * item_emb).sum(dim=1)

    def bpr_loss(self, user_idx, pos_idx, neg_idx):
        """
        BPR loss using current NGCF embeddings.

        user_idx : (B,)
        pos_idx  : (B,)
        neg_idx  : (B,)
        """
        E_user, E_item = self.propagate()

        u_e   = E_user[user_idx]     # (B, D')
        pos_e = E_item[pos_idx]      # (B, D')
        neg_e = E_item[neg_idx]      # (B, D')

        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)

        mf_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        reg_loss = (
            u_e.norm(2).pow(2)
            + pos_e.norm(2).pow(2)
            + neg_e.norm(2).pow(2)
        ) / user_idx.size(0)

        loss = mf_loss + self.l2_reg * reg_loss
        return loss # , mf_loss.detach(), reg_loss.detach()


def evaluator(
    model,
    data_loader,
    k: int,
    device,
    num_neg: int = 100,
    user_sample_size: int = 10000,
    is_test: bool = True
):
    """
    Evaulate NGCF model with leave-one-out style ranking:
    - For each user with at least one test interaction:
      - Choose one test item as the target.
      - Sample `num_neg` negative items.
      - Rank target among negatives and compute NDCG@k, Hit@k.

    Returns
    -------
    ndcg : float
    hit  : float
    """
    user_num, item_num = data_loader.user_num, data_loader.item_num

    # build train and test dicts: user -> list of items
    train_user_pos = data_loader.train_user_pos  # already a dict(user -> set(items))
    if is_test:
        test_user_pos = data_loader.test_user_pos
    else:
        test_user_pos = data_loader.val_user_pos

    # candidate users: those with at least one test item
    all_users = list(test_user_pos.keys())
    if len(all_users) == 0:
        return 0.0, 0.0

    if len(all_users) > user_sample_size:
        users = np.random.choice(all_users, size=user_sample_size, replace=False)
    else:
        users = all_users

    NDCG = 0.0
    HIT = 0.0
    valid_user = 0

    all_items = np.arange(item_num)

    model.eval()
    with torch.no_grad():
        for u in users:
            test_items = test_user_pos.get(u, [])
            if len(test_items) == 0:
                continue

            # pick one target test item (leave-one-out 스타일)
            target = np.random.choice(test_items)

            # items already interacted with (train + other test)
            rated = set(train_user_pos.get(u, []))
            rated.update(test_items)

            # sample negatives
            neg_items = []
            while len(neg_items) < num_neg:
                j = np.random.randint(0, item_num)
                if j not in rated and j not in neg_items:
                    neg_items.append(j)

            # candidate item list = [target] + negatives
            item_idx = np.array([target] + neg_items, dtype=np.int64)
            user_idx = np.full_like(item_idx, fill_value=u)

            user_tensor = torch.LongTensor(user_idx).to(device)
            item_tensor = torch.LongTensor(item_idx).to(device)

            scores = model.predict(user_tensor, item_tensor)  # (1 + num_neg,)
            scores = scores.detach().cpu().numpy()

            # rank target (index 0) among all candidates (higher score is better)
            rank = (-scores).argsort().tolist().index(0)

            valid_user += 1

            if rank < k:
                HIT += 1
                NDCG += 1 / np.log2(rank + 2)

    if valid_user == 0:
        return 0.0, 0.0

    ndcg = NDCG / valid_user
    hit = HIT / valid_user
    return ndcg, hit

def trainer(
    model,
    data_loader,
    optimizer,
    batch_size: int,
    epoch_num: int,
    num_batches_per_epoch: int,
    eval_interval: int,
    eval_k: int,
    patience,
    best_model_path,
    device
):
    """
    Training loop for NGCF with BPR Loss
    """

    best_ndcg = float("-inf")
    best_hit = 0.0
    epochs_without_improve = 0

    for epoch in range(1, epoch_num + 1):
        model.train()
        epoch_loss = 0.0
        
        for _ in range(num_batches_per_epoch):
            users, pos_items, neg_items = data_loader.get_bpr_batch(batch_size)
            
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()

            loss = model.bpr_loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches_per_epoch

        print(
            f"[Epoch {epoch}]"
            f"Train Loss: {avg_loss:.4f}"
        )

        # evaluate & early stopping
        if epoch % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                ndcg, hit = evaluator(
                    model,
                    data_loader,
                    eval_k,
                    device, 
                    is_test = False                    
                )

            print(f"Eval - NDCG@{eval_k}: {ndcg:.4f}, Hit@{eval_k}: {hit:.4f}")

            # check improvement
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_hit = hit

                torch.save(model.state_dict(), best_model_path)
                print(f"  ** Best model updated and saved to '{best_model_path}' **")

                epochs_without_improve = 0
            
            else:
                epochs_without_improve += 1
                print(f"  No improvement. Patience: {epochs_without_improve}/{patience}")

                if epochs_without_improve >= patience:
                    print("  >>> Early stopping triggered.")
                    break

    print("========================================")
    print(f"Best Eval : NDCG@{eval_k}={best_ndcg:.4f}, Hit@{eval_k}={best_hit:.4f}")
    print(f"Best model weights saved at: {best_model_path}")
