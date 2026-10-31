"""One implementation shared by the notebook and command-line entry point."""
from copy import deepcopy
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from . import baselines
from .data import Index, load_data, split_tail, relevant, popularity, recall_at_k, validate_predictions, write_submission, sha256

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    return json.loads(Path(path or ROOT / 'config.json').read_text(encoding='utf-8'))


def environment():
    import platform
    result = {'python': platform.python_version()}
    for name in ['numpy', 'pandas', 'scipy', 'matplotlib', 'threadpoolctl', 'replay-rec', 'torch', 'lightning',
                 'torchmetrics', 'pytorch-optimizer', 'polars', 'scikit-learn', 'pyarrow']:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    import sys
    if 'torch' in sys.modules:
        torch = sys.modules['torch']
        result['cuda_runtime'] = str(torch.version.cuda)
        result['device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def prepare(data, config):
    index = Index(data)
    known, held_out = split_tail(data.events, config['validation_fraction'])
    train = relevant(known, config['train_min_rating'])
    valid = relevant(held_out, config['target_min_rating'])
    targets = valid.groupby('user_id').item_id.agg(set).to_dict()
    return index, known, train, targets


def run_baselines(data_dir, output_dir, config):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(data_dir)
    index, known, train, targets = prepare(data, config)
    matrix, seen = index.matrix(train), index.matrix(known)
    users = sorted(targets)
    rows = []

    def evaluate(name, weights, parameters, started):
        predictions = baselines.recommend(index, matrix, seen, users, weights, config['k'],
                                          config['batch_size'], config['exclude_seen'])
        validate_predictions(predictions, users, index, known, config['k'], config['exclude_seen'])
        score, _ = recall_at_k(predictions, targets, config['k'])
        rows.append({'model': name, 'parameters_json': json.dumps(parameters, sort_keys=True),
                     'local_recall_at_10': score, 'evaluated_users': len(targets),
                     'elapsed_seconds': round(time.perf_counter() - started, 3)})
        print(f'{name}: Recall@10={score:.6f}', flush=True)

    started = time.perf_counter()
    evaluate('popularity', None, {}, started)
    started = time.perf_counter()
    weights = baselines.fit_itemknn(matrix, config['knn_neighbors'], config['knn_shrinkage'], config['batch_size'])
    evaluate('itemknn', weights, {'neighbors': config['knn_neighbors'], 'shrinkage': config['knn_shrinkage']}, started)
    del weights
    x = matrix.astype(np.float64)
    gram = (x.T @ x).toarray()
    del x
    for regularization in config['ease_regularization_grid']:
        started = time.perf_counter()
        weights = baselines.fit_ease(matrix, regularization, gram, config['blas_threads'])
        evaluate(f'ease_l2_{regularization:g}', weights, {'regularization': regularization}, started)
        del weights
    del gram
    metrics = pd.DataFrame(rows).sort_values(['local_recall_at_10', 'model'], ascending=[False, True], kind='stable')
    metrics.to_csv(output / 'metrics.csv', index=False)
    best = metrics.iloc[0]
    best_name, best_parameters = best['model'], json.loads(best['parameters_json'])
    full_matrix = index.matrix(relevant(data.events, config['train_min_rating']))
    if best_name.startswith('ease_l2_'):
        full_weights = baselines.fit_ease(full_matrix, best_parameters['regularization'], threads=config['blas_threads'])
        model_family = 'ease'
    elif best_name == 'itemknn':
        full_weights = baselines.fit_itemknn(full_matrix, config['knn_neighbors'], config['knn_shrinkage'])
        model_family = 'itemknn'
    else:
        full_weights, model_family = None, 'popularity'
    predictions = baselines.recommend(index, full_matrix, index.matrix(data.events), data.sample.user_id,
                                      full_weights, config['k'], config['batch_size'], config['exclude_seen'])
    destination = write_submission(predictions, data, index, output / f'submission_{model_family}.csv',
                                   config['k'], config['exclude_seen'])
    shutil.copyfile(destination, output / 'submission.csv')
    report = {'mode': 'baselines', 'selected_model': best_name, 'selected_parameters': best_parameters,
              'config': config, 'data_hashes': data.hashes, 'environment': environment(),
              'validation_results': metrics.to_dict(orient='records'), 'submission': destination.name,
              'submission_sha256': sha256(destination)}
    write_json(output / 'run_report.json', report)
    return report


def predict(data_dir, output_dir, config, model='sasrec', checkpoint=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(data_dir)
    if model == 'sasrec':
        from . import sasrec
        import torch
        torch.set_num_threads(config['blas_threads'])
        destination = Path(checkpoint) if checkpoint else output / 'checkpoints' / 'sasrec.pt'
        if not destination.is_file():
            raise FileNotFoundError(f'{destination} is missing. Copy sasrec_full.pt from Kaggle, '
                                    'or run: python run.py train')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        network, payload = sasrec.load_checkpoint(destination, device, data.hashes, expected_split='full')
        index = Index(data, payload['item_ids'])
        # Legacy checkpoints use the original default preprocessing; new checkpoints retain it explicitly.
        data_config = payload.get('metadata', {}).get('data_config', config)
        history_events = relevant(data.events, data_config['train_min_rating'])
        histories = index.histories(history_events)
        matrix = index.matrix(history_events)
        predictions = sasrec.recommend(
            network, index, histories, data.sample.user_id.tolist(), index.matrix(data.events), popularity(matrix),
            payload['parameters'], sasrec.trained_item_mask(histories, len(index.items)), device,
            data_config['k'], data_config['exclude_seen'],
        )
        config = data_config
        details = {'checkpoint': destination.name, 'checkpoint_sha256': sha256(destination), 'epoch': payload['epoch']}
    elif model == 'ease':
        index = Index(data)
        matrix = index.matrix(relevant(data.events, config['train_min_rating']))
        weights = baselines.fit_ease(matrix, config['ease_final_regularization'], threads=config['blas_threads'])
        predictions = baselines.recommend(index, matrix, index.matrix(data.events), data.sample.user_id,
                                           weights, config['k'], config['batch_size'], config['exclude_seen'])
        details = {'regularization': config['ease_final_regularization']}
    else:
        raise ValueError('Prediction model must be sasrec or ease')
    path = write_submission(predictions, data, index, output / f'submission_{model}.csv', config['k'], config['exclude_seen'])
    shutil.copyfile(path, output / 'submission.csv')
    report = {'mode': 'predict', 'selected_model': model, **details, 'data_hashes': data.hashes,
              'environment': environment(), 'submission': path.name, 'submission_sha256': sha256(path)}
    write_json(output / 'prediction_report.json', report)
    return report


def train(data_dir, output_dir, config, smoke=False):
    from . import sasrec
    import torch
    output = Path(output_dir) / 'smoke' if smoke else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(data_dir)
    index, known, training, targets = prepare(data, config)
    parameters = deepcopy(config['sasrec'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if parameters['require_cuda'] and device != 'cuda' and not smoke:
        raise RuntimeError('Enable a Kaggle GPU, or set sasrec.require_cuda=false for CPU training')
    torch.set_num_threads(config['blas_threads'])
    if not smoke:
        base_report = run_baselines(data_dir, output, config)
    histories = index.histories(training)
    eval_users = sorted(targets)
    if smoke:
        parameters.update({'max_sequence_length': 12, 'window_stride': 6, 'embedding_dim': 16,
                           'num_heads': 1, 'num_blocks': 1, 'batch_size': 16, 'prediction_batch_size': 32,
                           'max_epochs': 1, 'patience': 1, 'mixed_precision': False,
                           'require_cuda': False, 'max_steps_per_epoch': 2})
        histories = {uid: histories[uid] for uid in sorted(histories)[:32]}
        eval_users = sorted(set(histories) & set(targets))
    seen, matrix = index.matrix(known), index.matrix(training)
    counts = popularity(matrix)
    mask = sasrec.trained_item_mask(histories, len(index.items))

    def evaluate(network):
        predictions = sasrec.recommend(network, index, histories, eval_users, seen, counts, parameters,
                                        mask, device, config['k'], config['exclude_seen'])
        validate_predictions(predictions, eval_users, index, known, config['k'], config['exclude_seen'])
        return recall_at_k(predictions, {uid: targets[uid] for uid in eval_users}, config['k'])[0]

    common = {'data_hashes': data.hashes, 'data_config': config, 'environment': environment()}
    checkpoint_dir = output / 'checkpoints'
    started = time.perf_counter()
    network, history, best_epoch = sasrec.fit(
        histories, parameters, index, parameters['max_epochs'], checkpoint_dir / 'sasrec_validation_best.pt',
        config['seed'], {**common, 'training_split': 'train'}, evaluate, device,
    )
    score = evaluate(network)
    if smoke:
        report = {'mode': 'smoke', 'test_only': True, 'checkpoint_roundtrip': 'passed',
                  'best_epoch': best_epoch, 'users': len(eval_users), 'submission_created': False}
        write_json(output / 'smoke_report.json', report)
        return report
    metrics = pd.read_csv(output / 'metrics.csv')
    nn_row = pd.DataFrame([{'model': 'sasrec', 'parameters_json': json.dumps({**parameters, 'best_epoch': best_epoch}),
                            'local_recall_at_10': score, 'evaluated_users': len(targets),
                            'elapsed_seconds': time.perf_counter() - started}])
    metrics = pd.concat([metrics, nn_row], ignore_index=True).sort_values(
        ['local_recall_at_10', 'model'], ascending=[False, True], kind='stable')
    metrics.to_csv(output / 'metrics.csv', index=False)
    history.to_csv(output / 'sasrec_validation_history.csv', index=False)
    del network
    if device == 'cuda':
        torch.cuda.empty_cache()
    full_histories = index.histories(relevant(data.events, config['train_min_rating']))
    full_model, full_history, _ = sasrec.fit(
        full_histories, parameters, index, best_epoch, checkpoint_dir / 'sasrec.pt', config['seed'],
        {**common, 'training_split': 'full'}, None, device,
    )
    del full_model
    full_history.to_csv(output / 'sasrec_full_history.csv', index=False)
    predict(data_dir, output, config, 'sasrec')
    selected = metrics.iloc[0]['model']
    if selected != 'sasrec':
        source = output / base_report['submission']
        shutil.copyfile(source, output / 'submission.csv')
    report = {**base_report, 'mode': 'train', 'selected_model': selected,
              'selected_parameters': json.loads(metrics.iloc[0]['parameters_json']),
              'sasrec_best_epoch': best_epoch, 'validation_results': metrics.to_dict(orient='records'),
              'submission': 'submission.csv', 'submission_sha256': sha256(output / 'submission.csv')}
    write_json(output / 'run_report.json', report)
    return report


def evaluate_checkpoint(data_dir, output_dir, config, checkpoint=None):
    from . import sasrec
    import torch
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(data_dir)
    path = Path(checkpoint) if checkpoint else output / 'checkpoints' / 'sasrec_validation_best.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(config['blas_threads'])
    model, payload = sasrec.load_checkpoint(path, device, data.hashes, expected_split='train')
    config = payload.get('metadata', {}).get('data_config', config)
    _, known, training, targets = prepare(data, config)
    index = Index(data, payload['item_ids'])
    histories = index.histories(training)
    predictions = sasrec.recommend(
        model, index, histories, sorted(targets), index.matrix(known), popularity(index.matrix(training)),
        payload['parameters'], sasrec.trained_item_mask(histories, len(index.items)), device,
        config['k'], config['exclude_seen'],
    )
    validate_predictions(predictions, sorted(targets), index, known, config['k'], config['exclude_seen'])
    score, users = recall_at_k(predictions, targets, config['k'])
    users.to_csv(output / 'sasrec_validation_per_user.csv', index=False)
    report = {'mode': 'evaluate', 'selected_model': 'sasrec', 'local_recall_at_10': score,
              'epoch': payload['epoch'], 'checkpoint_sha256': sha256(path), 'data_hashes': data.hashes,
              'environment': environment()}
    write_json(output / 'sasrec_checkpoint_evaluation.json', report)
    print(f'SASRec checkpoint: epoch={payload["epoch"]}; Recall@10={score:.6f}', flush=True)
    return report


def run(mode='auto', model='sasrec', data_dir=None, output_dir=None, config=None, checkpoint=None, smoke=False):
    config = deepcopy(config or load_config())
    data_dir = Path(data_dir or ROOT / 'data')
    output = Path(output_dir or ROOT / 'outputs')
    if mode == 'auto':
        mode = 'predict' if checkpoint is not None or (output / 'checkpoints' / 'sasrec.pt').is_file() else 'baselines'
        print(f'Auto mode: {mode}. SASRec checkpoint: {"provided" if mode == "predict" else "not added"}', flush=True)
    if mode == 'baselines':
        return run_baselines(data_dir, output, config)
    if mode == 'predict':
        return predict(data_dir, output, config, model, checkpoint)
    if mode == 'train':
        return train(data_dir, output, config, smoke)
    if mode == 'evaluate':
        return evaluate_checkpoint(data_dir, output, config, checkpoint)
    raise ValueError(f'Unknown mode: {mode}')
