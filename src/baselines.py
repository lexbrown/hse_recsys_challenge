"""Popularity, ItemKNN and EASE with the same ranking rules."""
import numpy as np
from scipy import sparse, linalg
from threadpoolctl import threadpool_limits

from .data import popularity, top_k, user_seen


def fit_itemknn(matrix, neighbors=100, shrinkage=10, block_size=256):
    counts = popularity(matrix)
    n_items = matrix.shape[1]
    blocks = []
    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)
        cooc = (matrix[:, start:end].T @ matrix).tocoo()
        denominator = np.sqrt(counts[cooc.row + start] * counts[cooc.col])
        values = np.divide(cooc.data, denominator, out=np.zeros_like(cooc.data), where=denominator > 0)
        if shrinkage:
            values *= cooc.data / (cooc.data + shrinkage)
        values[cooc.row + start == cooc.col] = 0
        block = sparse.csr_matrix((values, (cooc.row, cooc.col)), shape=(end - start, n_items))
        block.eliminate_zeros()
        for row in range(end - start):
            lo, hi = block.indptr[row:row + 2]
            weights, indices = block.data[lo:hi], block.indices[lo:hi]
            if len(weights) > neighbors:
                boundary = np.partition(weights, -neighbors)[-neighbors]
                candidates = np.flatnonzero(weights >= boundary)
                selected = candidates[np.lexsort((indices[candidates], -weights[candidates]))[:neighbors]]
                keep = np.zeros(len(weights), dtype=bool)
                keep[selected] = True
                weights[~keep] = 0
        block.eliminate_zeros()
        blocks.append(block)
    return sparse.vstack(blocks, format='csr')


def fit_ease(matrix, regularization=1000, gram=None, threads=4):
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('EASE regularization must be positive and finite')
    n_items = matrix.shape[1]
    if gram is None:
        x = matrix.astype(np.float64)
        gram = (x.T @ x).toarray()
    system = np.array(gram, dtype=np.float64, order='F', copy=True)
    system[np.diag_indices(n_items)] += regularization
    with threadpool_limits(limits=threads, user_api='blas'):
        factor = linalg.cho_factor(system, lower=True, overwrite_a=True, check_finite=False)
        inverse = linalg.cho_solve(factor, np.eye(n_items), check_finite=False)
    inverse /= -inverse.diagonal().copy()[None, :]
    np.fill_diagonal(inverse, 0)
    if not np.isfinite(inverse).all():
        raise ArithmeticError('Non-finite EASE weights')
    return inverse.astype(np.float32)


def recommend(index, matrix, seen, users, weights=None, k=10, batch_size=256, exclude_seen=True):
    counts = popularity(matrix)
    result = {}
    users = list(users)
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        rows = np.array([index.user_map[uid] for uid in batch])
        if weights is None:
            scores = np.broadcast_to(counts, (len(rows), len(index.items)))
        else:
            product = matrix[rows] @ weights
            scores = product.toarray() if sparse.issparse(product) else np.asarray(product)
        for offset, (uid, row) in enumerate(zip(batch, rows)):
            result[uid] = top_k(scores[offset], counts, user_seen(seen, row), index.items, k, exclude_seen)
    return result
