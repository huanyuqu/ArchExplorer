"""Generate dimensions-only scale requests; this never executes large models."""
import argparse
import json
from pathlib import Path


def main():
    from arch_explorer.archs.paper_scale import llm_scale_cases, mla_scale_cases, export_scale_case
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('llm', 'mla', 'export'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=('fp32', 'fp16', 'bf16'))
    parser.add_argument('--directory')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--seq-len', type=int)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--width', type=int)
    parser.add_argument('--query-rank', type=int)
    parser.add_argument('--content-dim', type=int)
    parser.add_argument('--output-width', type=int)
    args = parser.parse_args()
    if args.kind == 'llm':
        cases = llm_scale_cases(batch=args.batch, seq_len=args.seq_len, dtype=args.dtype, tp=args.tp)
    elif args.kind == 'mla':
        cases = mla_scale_cases(width=args.width, query_rank=args.query_rank, content_dim=args.content_dim,
                                output_width=args.output_width, dtype=args.dtype)
    else:
        if args.directory is None:
            parser.error('export requires --directory')
        cases = [export_scale_case(args.directory)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'schema': 'arch-explorer.scale-cases.v1',
                                  'executed': False, 'cases': cases}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
