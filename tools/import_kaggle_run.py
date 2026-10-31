"""Import the REAL full-data checkpoint and reports from the original Kaggle notebook."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Unpacked old Kaggle outputs directory')
    parser.add_argument('--checkpoint', help='Explicit full-data checkpoint if several runs exist')
    parser.add_argument('--kaggle-score', type=float, help='Actual score of this SASRec submission, if known')
    args = parser.parse_args()
    source = Path(args.source).resolve()
    candidates = [Path(args.checkpoint).resolve()] if args.checkpoint else sorted(source.rglob('sasrec_full.pt'))
    if len(candidates) != 1 or not candidates[0].is_file():
        raise SystemExit(f'Expected one sasrec_full.pt; found {len(candidates)}. '
                         'Finish the original notebook section 12, or pass --checkpoint explicitly.')
    checkpoint = candidates[0]
    if checkpoint.name not in {'sasrec_full.pt', 'sasrec.pt'}:
        raise SystemExit('Use the FULL-data checkpoint, not sasrec_validation_best.pt')
    import torch
    from src.data import load_data, sha256
    from src.sasrec import create_model
    data = load_data(ROOT / 'data')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload['parameters'].get('max_steps_per_epoch') is not None:
        raise SystemExit('This is a smoke/test checkpoint, not a full training run')
    if set(payload['item_ids']) != set(data.items.item_id):
        raise SystemExit('Checkpoint catalog differs from this project data')
    model = create_model(payload['parameters'], len(payload['item_ids']))
    model.load_state_dict(payload['model_state'], strict=True)
    destination = ROOT / 'outputs' / 'checkpoints'
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, destination / 'sasrec.pt')
    copies = {'sasrec_full.pt': sha256(checkpoint)}
    for name, new_name in [
        ('sasrec_validation_best.pt', 'sasrec_validation_best.pt'),
        ('validation_history.csv', '../sasrec_validation_history.csv'),
        ('sasrec_full.history.csv', '../sasrec_full_history.csv'),
    ]:
        original = checkpoint.parent / name
        if original.is_file():
            target = (destination / new_name).resolve()
            shutil.copyfile(original, target)
            copies[name] = sha256(original)
    submissions = list(source.rglob('submission_sasrec.csv'))
    if len(submissions) == 1:
        shutil.copyfile(submissions[0], ROOT / 'outputs' / 'submission_sasrec.csv')
        copies['submission_sasrec.csv'] = sha256(submissions[0])
    history_path = ROOT / 'outputs' / 'sasrec_validation_history.csv'
    best = None
    if history_path.exists():
        with history_path.open(encoding='utf-8', newline='') as handle:
            rows = [row for row in csv.DictReader(handle) if row.get('local_recall_at_10')]
        if rows:
            best = max(rows, key=lambda row: float(row['local_recall_at_10']))
    if best:
        import pandas as pd
        metrics_path = ROOT / 'outputs' / 'metrics.csv'
        metrics = pd.read_csv(metrics_path if metrics_path.exists() else ROOT / 'outputs/reference_metrics.csv')
        metrics = metrics.loc[metrics.model != 'sasrec']
        metrics = pd.concat([metrics, pd.DataFrame([{
            'model': 'sasrec', 'parameters_json': json.dumps(payload['parameters'], sort_keys=True),
            'local_recall_at_10': float(best['local_recall_at_10']),
            'evaluated_users': None, 'elapsed_seconds': sum(float(row['elapsed_seconds']) for row in rows),
        }])], ignore_index=True)
        metrics.sort_values('local_recall_at_10', ascending=False).to_csv(metrics_path, index=False)
    report = {'source_checkpoint': checkpoint.name, 'imported_sha256': copies,
              'checkpoint_epoch': int(payload['epoch']),
              'best_local_recall_at_10': float(best['local_recall_at_10']) if best else None,
              'best_validation_epoch': int(best['epoch']) if best else None,
              'parameters': payload['parameters'], 'environment': payload.get('versions', {}),
              'confirmed_kaggle_score': args.kaggle_score,
              'note': 'Legacy checkpoint is copied unchanged. No training-data hashes were invented.'}
    (ROOT / 'outputs' / 'imported_kaggle_run.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if best:
        import re
        readme = ROOT / 'README.md'
        text = readme.read_text(encoding='utf-8')
        public = str(args.kaggle_score) if args.kaggle_score is not None else '—'
        row = f'| SASRec | {float(best["local_recall_at_10"]):.6f} | {public} |'
        text = re.sub(r'^\| SASRec \|.*$', row, text, flags=re.MULTILINE)
        text = re.sub(r'¹ Наблюдение.*?это не метрика тестового CPU-прогона\.',
                      f'Для SASRec использована сохранённая история: лучшая эпоха {int(best["epoch"])}; '
                      f'финальная модель обучалась {int(payload["epoch"])} эпох.', text, flags=re.DOTALL)
        readme.write_text(text, encoding='utf-8')
        journal = ROOT / 'EXPERIMENTS.md'
        text = journal.read_text(encoding='utf-8')
        block = (f'Импортирован GPU-запуск: лучший локальный Recall@10 = **{float(best["local_recall_at_10"]):.6f}**, '
                 f'лучшая эпоха — **{int(best["epoch"])}**. Полный checkpoint использует '
                 f'**{int(payload["epoch"])}** эпох. Kaggle score: **{public}**. '
                 'Контрольные суммы и параметры сохранены в `outputs/imported_kaggle_run.json`.')
        start, end = '<!-- SASREC_RESULT_START -->', '<!-- SASREC_RESULT_END -->'
        text = re.sub(re.escape(start) + r'.*?' + re.escape(end), start + '\n' + block + '\n' + end,
                      text, flags=re.DOTALL)
        if args.kaggle_score is not None:
            text = re.sub(r'^\| SASRec \| `submission_sasrec.csv`.*$',
                          f'| SASRec | `submission_sasrec.csv` | {args.kaggle_score} | Реальный сабмит импортированного запуска |',
                          text, flags=re.MULTILINE)
        journal.write_text(text, encoding='utf-8')
    print('Imported:', destination / 'sasrec.pt')
    print('Now run: python run.py predict --model sasrec')


if __name__ == '__main__':
    main()
