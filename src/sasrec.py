"""RePlay SASRec, causal windows, checkpointing and inference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from pathlib import Path
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from replay.data import FeatureHint, FeatureSource, FeatureType
from replay.data.nn import TensorSchema, TensorFeatureInfo, TensorFeatureSource
from replay.nn.sequential import SasRec

from .data import popularity, top_k, user_seen


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class Windows(Dataset):
    def __init__(self, histories, max_length, stride, padding):
        if max_length < 1 or not 1 <= stride <= max_length:
            raise ValueError('Window stride must be in [1, max_length]')
        self.histories, self.max_length, self.padding = histories, max_length, padding
        self.windows = []
        for uid in sorted(histories):
            size = len(histories[uid])
            if size >= 2:
                ends = list(range(max_length + 1, size, stride)) + [size]
                self.windows.extend((uid, end) for end in ends)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, position):
        uid, end = self.windows[position]
        sequence = self.histories[uid][max(0, end - self.max_length - 1):end]
        padded = np.full(self.max_length + 1, self.padding, dtype=np.int64)
        padded[-len(sequence):] = sequence
        inputs, labels = padded[:-1].copy(), padded[1:].copy()
        target_mask = (inputs != self.padding) & (labels != self.padding)
        return torch.from_numpy(inputs), torch.from_numpy(labels), torch.from_numpy(target_mask)


def create_model(parameters, n_items):
    schema = TensorSchema([TensorFeatureInfo(
        name='item_id', feature_type=FeatureType.CATEGORICAL, is_seq=True,
        feature_hint=FeatureHint.ITEM_ID, cardinality=n_items, padding_value=n_items,
        embedding_dim=parameters['embedding_dim'],
        feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, 'item_id')],
    )])
    return SasRec.from_params(
        schema=schema, embedding_dim=parameters['embedding_dim'], num_heads=parameters['num_heads'],
        num_blocks=parameters['num_blocks'], max_sequence_length=parameters['max_sequence_length'],
        dropout=parameters['dropout'],
    )


def save_checkpoint(model, parameters, epoch, item_ids, destination, metadata=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format_version': 1,
        'model_state': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        'parameters': dict(parameters), 'epoch': int(epoch), 'item_ids': list(item_ids),
        'metadata': metadata or {},
    }
    temporary = destination.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(path, device='cpu', expected_hashes=None, expected_split=None):
    """Also accepts sasrec_full.pt / sasrec_validation_best.pt from the original notebook."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    required = {'model_state', 'parameters', 'epoch', 'item_ids'}
    if not required <= payload.keys():
        raise ValueError(f'Checkpoint lacks required keys: {required - payload.keys()}')
    metadata = payload.get('metadata', {})
    if expected_hashes and metadata.get('data_hashes') and metadata['data_hashes'] != expected_hashes:
        raise ValueError('Data fingerprints differ from the checkpoint training data')
    if expected_split and metadata.get('training_split') not in (None, expected_split):
        raise ValueError(f'Expected a {expected_split} checkpoint, got {metadata.get("training_split")}')
    if not metadata:
        is_validation = 'validation' in Path(path).name
        if expected_split == 'train' and not is_validation:
            raise ValueError('For local evaluation use sasrec_validation_best.pt, not the full-data model')
        if expected_split == 'full' and is_validation:
            raise ValueError('For a submission use the full-data model, not the validation checkpoint')
        warnings.warn('Legacy notebook checkpoint: catalog IDs are checked, but data fingerprints '
                      'and training split were not stored inside this file.', stacklevel=2)
    model = create_model(payload['parameters'], len(payload['item_ids']))
    model.load_state_dict(payload['model_state'], strict=True)
    return model.to(device).eval(), payload


@torch.inference_mode()
def recommend(model, index, histories, requested_users, seen, counts, parameters,
              trained_items, device='cpu', k=10, exclude_seen=True):
    model.eval()
    n_items = len(index.items)
    result = {}
    known = [uid for uid in requested_users if len(histories.get(uid, []))]
    missing = [uid for uid in requested_users if not len(histories.get(uid, []))]
    length = parameters['max_sequence_length']
    for start in range(0, len(known), parameters['prediction_batch_size']):
        users = known[start:start + parameters['prediction_batch_size']]
        inputs = np.full((len(users), length), n_items, dtype=np.int64)
        for row, uid in enumerate(users):
            tail = histories[uid][-length:]
            inputs[row, -len(tail):] = tail
        tokens = torch.as_tensor(inputs, device=device)
        logits = model(feature_tensors={'item_id': tokens}, padding_mask=tokens.ne(n_items))['logits']
        scores = logits.float().cpu().numpy()
        if scores.shape != (len(users), n_items) or not np.isfinite(scores).all():
            raise ArithmeticError('Invalid SASRec logits')
        scores[:, ~trained_items] = -np.inf
        for row, uid in enumerate(users):
            result[uid] = top_k(scores[row], counts, user_seen(seen, index.user_map[uid]),
                                index.items, k, exclude_seen)
    for uid in missing:
        result[uid] = top_k(counts, counts, user_seen(seen, index.user_map[uid]), index.items, k, exclude_seen)
    return result


def fit(histories, parameters, index, epochs, destination, seed=42,
        metadata=None, validation=None, device='cpu'):
    seed_everything(seed)
    padding = len(index.items)
    model = create_model(parameters, padding).to(device)
    dataset = Windows(histories, parameters['max_sequence_length'], parameters['window_stride'], padding)
    if not len(dataset):
        raise ValueError('No training histories of length >= 2')
    loader = DataLoader(dataset, batch_size=parameters['batch_size'], shuffle=True,
                        num_workers=parameters.get('num_workers', 0),
                        generator=torch.Generator().manual_seed(seed),
                        pin_memory=str(device).startswith('cuda'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['learning_rate'],
                                   weight_decay=parameters['weight_decay'])
    amp = str(device).startswith('cuda') and parameters['mixed_precision']
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, best_score, stale = [], -np.inf, 0
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum, targets = 0.0, 0
        for step, (tokens, labels, target_mask) in enumerate(loader):
            if step >= parameters.get('max_steps_per_epoch', len(loader)):
                break
            tokens, labels, target_mask = [x.to(device, non_blocking=True) for x in (tokens, labels, target_mask)]
            count = int(target_mask.sum().item())
            if not count:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
                loss = model(feature_tensors={'item_id': tokens}, padding_mask=tokens.ne(padding),
                             positive_labels=labels.unsqueeze(-1), target_padding_mask=target_mask.unsqueeze(-1))['loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['gradient_clip'], error_if_nonfinite=not amp)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().item()) * count
            targets += count
        if not targets:
            raise RuntimeError('No optimization steps in this epoch')
        score = validation(model) if validation else None
        history.append({'epoch': epoch, 'train_loss': loss_sum / targets, 'local_recall_at_10': score,
                        'elapsed_seconds': time.perf_counter() - started})
        pd.DataFrame(history).to_csv(destination.with_suffix('.history.csv'), index=False)
        if validation is None or score > best_score:
            best_score = score if score is not None else best_score
            stale = 0
            save_checkpoint(model, parameters, epoch, index.items.tolist(), destination,
                            {**(metadata or {}), 'seed': seed})
        else:
            stale += 1
        text = f'{score:.6f}' if score is not None else 'full-data refit'
        print(f'Epoch {epoch:02d}/{epochs}: loss={loss_sum / targets:.4f}; Recall@10={text}; '
              f'{history[-1]["elapsed_seconds"]:.1f}s', flush=True)
        if validation is not None and stale >= parameters['patience']:
            print('Early stopping', flush=True)
            break
    del optimizer, scaler, model
    model, payload = load_checkpoint(destination, device)
    return model, pd.DataFrame(history), payload['epoch']


def trained_item_mask(histories, n_items):
    mask = np.zeros(n_items, dtype=bool)
    for sequence in histories.values():
        mask[sequence] = True
    return mask
