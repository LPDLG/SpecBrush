"""Evaluation helper for the released Stage-I release."""
from __future__ import annotations
import argparse, json, torch
from inference.pipeline import evaluate_test, load_checkpoint


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--test_npz',required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--condition',choices=['rgb_only','full'],default='rgb_only')
    ap.add_argument('--num_samples',type=int,default=20)
    ap.add_argument('--no_rts',action='store_true')
    args=ap.parse_args()
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    bundle=load_checkpoint(args.ckpt,device)
    result=evaluate_test(bundle,args.test_npz,device,condition=args.condition,num_samples=args.num_samples,apply_rts=not args.no_rts)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
