import argparse
from .config import Config


def main():
    parser=argparse.ArgumentParser(description='Fixed-cutoff tram forecasting')
    commands=parser.add_subparsers(dest='command',required=True)
    check=commands.add_parser('check-config',help='Validate resolved model configuration')
    check.add_argument('--config',default='configs/default.json')
    for name in ('prepare','inspect','smoke','train','evaluate','refit','predict'):
        p=commands.add_parser(name,help='Pending integration')
        p.add_argument('--config',default='configs/default.json')
        p.add_argument('--artifact')
        if name=='prepare': p.add_argument('--regime',choices=['validation','final'],required=True)
        if name=='train': p.add_argument('--model',choices=['boarding_only','full'],required=True)
        if name in ('evaluate','predict'): p.add_argument('--checkpoint',required=True)
        if name=='refit': p.add_argument('--selected-checkpoint',required=True)
        if name=='predict':
            p.add_argument('--template',required=True); p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='check-config':
        print(Config.load(args.config).fingerprint)
    else:
        parser.error(f'{args.command} is not implemented yet; consult implementation_progress.md')
