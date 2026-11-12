# Concrete Accel-Sim implementation; owned by the component provider.
"""Frozen finite-campaign worker with a recoverable supervisor stop signal."""
import argparse,signal
from arch_explorer.components.accelsim.microcampaign import Collector

def stop(signum,frame):
    raise TimeoutError('absolute resource supervisor requested stop; no further observations')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('design');p.add_argument('handoff');p.add_argument('window');a=p.parse_args(argv)
    for name in ('SIGALRM','SIGTERM'):signal.signal(getattr(signal,name),stop)
    from arch_explorer.components.accelsim.microbundle import json_file
    cls=Collector
    if json_file(a.design)[0].get('schema')=='arch-explorer.measurement-state-device-design.v1':
        from arch_explorer.components.accelsim.state_device import Campaign
        cls=Campaign
    error=cls(a.design,a.handoff,a.window).execute()
    return 1 if error is not None else 0

if __name__=='__main__':raise SystemExit(main())
