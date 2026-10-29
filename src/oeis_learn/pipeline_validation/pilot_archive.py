"""Bounded owned export calibration, no input data or private state in fixtures."""
import argparse
from pathlib import Path
import random
import shutil
import time
from .identity import seal,write,digest
from .bundle import pack,verify


def calibrate(output,source):
    output=Path(output);output.mkdir(parents=True)
    observations=[]
    for ordinal in range(3):
        root=output/('probe-'+str(ordinal));root.mkdir()
        started=time.monotonic();rng=random.Random(20261016+ordinal)
        write(root/'summary.json',{'selected_scope':'PASS','fixture':True})
        write(root/'contract.json',{'fixture':'public synthetic export cost; no learning evidence'})
        # 4094*4098 plus two compact core files is at least16MiB over4096 members.
        names=['summary.json','contract.json']
        for i in range(4094):
            name='details/'+str(i)+'.txt';path=root/name;path.parent.mkdir(exist_ok=True)
            path.write_bytes(rng.randbytes(4098));names.append(name)
        size=sum((root/n).stat().st_size for n in names)
        archive=pack(root,mandatory=names,cap=32<<20,optional_cap=0)
        manifest=verify(archive)
        elapsed=time.monotonic()-started
        observations.append(seal(dict(version='pipeline-pilot-archive-observation/v1',ordinal=ordinal,
            source=source,complete=not manifest['omissions'],seconds=elapsed,bytes=size,members=len(names),
            filesystem_device=root.stat().st_dev,fixture=True)))
        archive.unlink();shutil.rmtree(root)
    write(output/'observations.json',observations)
    return observations


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',required=True);args=p.parse_args(argv)
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    actual=digest(collect_snapshot(Path(__file__).resolve().parents[3]))
    if actual!=args.source:raise ValueError('archive calibration source mismatch')
    calibrate(args.output,args.source)


if __name__=='__main__':main()
