"""One public preflight row, suitable for a bounded scale-run child command."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    from arch_explorer.archs.paper_scale import preflight_scale_case
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',required=True)
    parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--resource-budget')
    parser.add_argument('--build-budget')
    parser.add_argument('--cache-replicas',type=int,default=0)
    parser.add_argument('--tiles')
    parser.add_argument('--streaming-plan', help='Explicit chunk/slots/capacity/cache_mode JSON; static estimate only')
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    value=json.loads(Path(args.cases).read_text())
    if value.get('schema')!='arch-explorer.scale-cases.v1' or type(value.get('cases')) is not list:
        raise ValueError('scale-cases.v1 with complete cases required')
    if not 0<=args.index<len(value['cases']):
        raise ValueError('case index is out of range')
    read=lambda p:json.loads(Path(p).read_text()) if p else None
    output=Path(args.output)
    if output.exists():
        raise ValueError('Preflight output must be new')
    record={'schema':'arch-explorer.scale-preflight.v1','case':value['cases'][args.index],'status':'started'}
    try:
        record=preflight_scale_case(record['case'],resource_budget=read(args.resource_budget),
            build_budget=read(args.build_budget),cache_replicas=args.cache_replicas,tiles=read(args.tiles),
            streaming=read(args.streaming_plan))
        return 0 if record['status'] in ('admitted','plan-estimated') else 2
    except BaseException as exc:
        record.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':record['status'],'output':str(output)}))


if __name__=='__main__':
    raise SystemExit(main())
