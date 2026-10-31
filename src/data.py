"""Data loading, chronological validation, ranking and submission checks."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

FILES = ('events.csv', 'user_features.csv', 'item_features.csv', 'submission_sample.csv')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Data:
    events: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame
    sample: pd.DataFrame
    hashes: dict

    @property
    def item_ids(self):
        return np.array(sorted(self.items.item_id), dtype=object)

    @property
    def user_ids(self):
        return np.array(sorted(self.users.user_id), dtype=object)

    def summary(self):
        lengths = self.events.groupby('user_id').size()
        return pd.Series({
            'events': len(self.events), 'users': len(self.users), 'catalog_items': len(self.items),
            'items_with_events': self.events.item_id.nunique(),
            'minimum_history': lengths.min(), 'median_history': lengths.median(),
        }, name='value').to_frame()


def load_data(directory):
    directory = Path(directory)
    missing = [name for name in FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing data files in {directory}: {missing}')
    frames = [pd.read_csv(directory / name, dtype={'user_id': str, 'item_id': str}) for name in FILES]
    events, users, items, sample = frames
    if set(events.columns) != {'user_id', 'item_id', 'rating', 'timestamp'}:
        raise ValueError('Unexpected events.csv schema')
    if list(sample.columns) != ['user_id', 'item_id']:
        raise ValueError('Unexpected submission_sample.csv schema')
    if any(frame.isna().any().any() for frame in frames):
        raise ValueError('Missing values in source data')
    if not events.rating.between(1, 5).all():
        raise ValueError('Ratings must be in [1, 5]')
    if not pd.api.types.is_numeric_dtype(events.timestamp):
        raise ValueError('timestamp must be a numeric within-user order')
    if events.duplicated(['user_id', 'item_id']).any():
        raise ValueError('Repeated user-item interactions need a separate policy')
    if events.duplicated(['user_id', 'timestamp']).any():
        raise ValueError('Tied within-user timestamps need a separate policy')
    if not users.user_id.is_unique or not items.item_id.is_unique or not sample.user_id.is_unique:
        raise ValueError('Duplicate IDs in catalog, user features or sample')
    if not set(events.user_id) <= set(users.user_id) or not set(sample.user_id) <= set(users.user_id):
        raise ValueError('Unknown user ID')
    if not set(events.item_id) <= set(items.item_id):
        raise ValueError('Unknown item ID')
    return Data(*frames, hashes={name: sha256(directory / name) for name in FILES})


def split_tail(events, fraction=0.2):
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must be between 0 and 1')
    ordered = events.sort_values(['user_id', 'timestamp'], kind='stable')
    sizes = ordered.groupby('user_id').item_id.transform('size')
    if (sizes < 2).any():
        raise ValueError('At least two events per user are required for this split')
    held_out = np.minimum(np.ceil(sizes * fraction).astype(int), sizes - 1)
    mask = ordered.groupby('user_id').cumcount() >= sizes - held_out
    train, valid = ordered.loc[~mask].copy(), ordered.loc[mask].copy()
    assert set(train.index).isdisjoint(valid.index)
    assert (train.groupby('user_id').timestamp.max() < valid.groupby('user_id').timestamp.min()).all()
    return train, valid


def relevant(events, threshold):
    return events if threshold is None else events.loc[events.rating >= threshold]


class Index:
    def __init__(self, data, item_ids=None):
        self.users = data.user_ids
        self.items = data.item_ids if item_ids is None else np.array(item_ids, dtype=object)
        if len(self.items) != len(set(self.items)) or set(self.items) != set(data.items.item_id):
            raise ValueError('Checkpoint catalog differs from item_features.csv')
        self.user_map = {uid: i for i, uid in enumerate(self.users)}
        self.item_map = {iid: i for i, iid in enumerate(self.items)}

    def matrix(self, events):
        rows = events.user_id.map(self.user_map)
        columns = events.item_id.map(self.item_map)
        if rows.isna().any() or columns.isna().any():
            raise ValueError('Unknown IDs while creating an interaction matrix')
        matrix = sparse.csr_matrix(
            (np.ones(len(events), dtype=np.float32), (rows.to_numpy(), columns.to_numpy())),
            shape=(len(self.users), len(self.items)), dtype=np.float32,
        )
        matrix.sum_duplicates()
        matrix.data[:] = 1
        return matrix

    def histories(self, events):
        ordered = events.sort_values(['user_id', 'timestamp'], kind='stable')
        return {uid: np.array([self.item_map[i] for i in group.item_id], dtype=np.int64)
                for uid, group in ordered.groupby('user_id', sort=False)}


def popularity(matrix):
    return np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)


def top_k(scores, counts, seen, item_ids, k=10, exclude_seen=True):
    allowed = np.ones(len(item_ids), dtype=bool)
    if exclude_seen:
        allowed[seen] = False
    candidates = np.flatnonzero(allowed)
    if len(candidates) < k:
        raise ValueError('Not enough candidate items to produce top-k')
    order = np.lexsort((candidates, -counts[candidates], -scores[candidates]))
    return item_ids[candidates[order[:k]]].tolist()


def user_seen(matrix, user_index):
    return matrix.indices[matrix.indptr[user_index]:matrix.indptr[user_index + 1]]


def recall_at_k(predictions, targets, k=10):
    rows = []
    for uid, actual in targets.items():
        if not actual:
            continue
        predicted = predictions.get(uid, [])[:k]
        if len(predicted) != len(set(predicted)):
            raise ValueError('Repeated items in recommendations')
        hits = len(set(predicted) & set(actual))
        rows.append((uid, hits, len(actual), hits / len(actual)))
    report = pd.DataFrame(rows, columns=['user_id', 'hits', 'target_count', 'recall'])
    if report.empty:
        raise ValueError('No users with non-empty validation targets')
    return float(report.recall.mean()), report


def validate_predictions(predictions, users, index, known_events, k=10, exclude_seen=True):
    if set(predictions) != set(users):
        raise ValueError('Prediction user IDs differ from requested IDs')
    catalog = set(index.items)
    seen = known_events.groupby('user_id').item_id.agg(set).to_dict()
    for uid, values in predictions.items():
        if len(values) != k or len(set(values)) != k or not set(values) <= catalog:
            raise ValueError(f'Invalid top-{k} recommendations for {uid}')
        if exclude_seen and set(values) & seen.get(uid, set()):
            raise ValueError(f'Seen items in recommendations for {uid}')


def write_submission(predictions, data, index, destination, k=10, exclude_seen=True):
    validate_predictions(predictions, data.sample.user_id, index, data.events, k, exclude_seen)
    result = data.sample[['user_id']].copy()
    result['item_id'] = result.user_id.map(lambda uid: ' '.join(predictions[uid]))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    pd.testing.assert_frame_equal(pd.read_csv(destination, dtype=str), result.reset_index(drop=True))
    return destination
