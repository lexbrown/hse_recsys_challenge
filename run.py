"""CLI: the same pipeline is called by recsys_start.ipynb."""
import argparse
from src.project import load_config, run


def main():
    parser = argparse.ArgumentParser(description='HSE RecSys: validate, train or predict')
    parser.add_argument('mode', choices=['auto', 'baselines', 'train', 'predict', 'evaluate'], nargs='?', default='auto')
    parser.add_argument('--model', choices=['sasrec', 'ease'], default='sasrec')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--checkpoint')
    parser.add_argument('--smoke', action='store_true', help='Short neural integration check, no submission')
    args = parser.parse_args()
    report = run(args.mode, args.model, args.data_dir, args.output_dir, load_config(args.config), args.checkpoint, args.smoke)
    print(f'Completed: {report["mode"]}; model: {report.get("selected_model", "test only")}')
    if report.get('submission'):
        print('Submission:', args.output_dir + '/' + report['submission'])


if __name__ == '__main__':
    main()
