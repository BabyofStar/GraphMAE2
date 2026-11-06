import argparse
from pathlib import Path
import protocol

parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,default=Path('/home/jovyan/work/GraphMAE2'))
args=parser.parse_args()
protocol.check_package()
result=protocol.prior_audit(args.repo.resolve())
print('PRIOR_P12_INDEPENDENT_AUDIT=PASS; '+result['directory'])
