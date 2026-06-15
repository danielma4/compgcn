"""No-message-passing ablation for a trained checkpoint.

Re-evaluates relation prediction with the graph's neighbour edges removed: an
empty edge set makes CompGCNConv fall back to its self-loop transform only, so
each node embedding is just its (trained) Brick-class feature passed through
the conv with **no neighbour aggregation**. Comparing this to the full graph
isolates how much accuracy comes from graph structure vs. type features alone.

Usage:
    python ablation.py --config config/brick.yaml \
        --load 'checkpoints/brick_distmult_corr_04_06_2026_10:11:54' --gpu -1
"""

import os
import torch

from helper import set_gpu
from run import build_parser, parse_with_yaml, Runner


def empty_graph(device):
    return (torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0,),   dtype=torch.long, device=device))


def main():
    parser = build_parser()
    parser.add_argument('--load', '-load', dest='load_path', default=None,
                        help='Checkpoint path (default: ./checkpoints/<name>)')
    args = parse_with_yaml(parser)

    set_gpu(args.gpu)
    runner = Runner(args)
    load_path = args.load_path or os.path.join('./checkpoints', args.name)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path} (pass --load <path>)")
    runner.load_model(load_path)

    print(f"\n## No-message-passing ablation (checkpoint: {load_path})\n")
    hdr = f"{'split':<6} {'setting':<12} {'MRR':>8} {'H@1':>8}"
    print(hdr); print('-' * len(hdr))

    graph_attr = {'valid': ('edge_index_valid', 'edge_type_valid'),
                  'test':  ('edge_index_test',  'edge_type_test')}
    for split in ['valid', 'test']:
        full = runner.predict_relation(split)

        ei_name, et_name = graph_attr[split]
        saved = (getattr(runner, ei_name), getattr(runner, et_name))
        setattr(runner, ei_name, empty_graph(runner.device)[0])
        setattr(runner, et_name, empty_graph(runner.device)[1])
        nostruct = runner.predict_relation(split)
        setattr(runner, ei_name, saved[0]); setattr(runner, et_name, saved[1])

        print(f"{split:<6} {'full graph':<12} {full['mrr']:>8.4f} {full['hits@1']:>8.4f}")
        print(f"{split:<6} {'type-only':<12} {nostruct['mrr']:>8.4f} {nostruct['hits@1']:>8.4f}")


if __name__ == '__main__':
    main()
