"""Offline shared-innovation authoring: python synthetic_innovations.py DIR TIER [--retail].

Each 32-byte block is SHA256(UTF8("creditratingsystem/reference-v1/STREAM/ROW/BLOCK")).
Digest bytes are written unchanged; consumers interpret uint32 little endian.
This portable counter construction is an engineering DGP, not a cryptographic API.
It intentionally differs from the fixed PCG64 draws in the packaged reference.
"""
import argparse
import hashlib
import json
from pathlib import Path


def stream_shapes(tier, retail=False):
    e,y,m,p,s,a,h,am={"tiny":(64,3,36,512,24,320,160,24),
        "standard":(3000,10,240,16384,60,50000,20000,60),
        "stress":(20000,15,360,131072,120,500000,200000,120)}[tier]
    shapes={"macro_history":[m,10],"macro_revision":[m,8],"macro_paths":[p*s,10],
        "obligor_static":[e,16],"obligor_year":[e*y,16],"obligor_groups":[((e+3)//4)*y,2],
        "facilities":[e,16],"country_transfer":[6*y,2]}
    if retail: shapes.update(retail_households=[h,16],retail_apps=[a,24],retail_account=[a*am,8])
    return shapes


def write_innovations(directory,tier,retail=False,experiment_id="reference-v1"):
    root=Path(directory)
    if not isinstance(experiment_id,str) or not experiment_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in experiment_id):
        raise ValueError("experiment_id requires ASCII letters, numbers, hyphen or underscore")
    shapes=stream_shapes(tier,retail)
    if root.exists() and any(root.iterdir()): raise ValueError("output must be absent or empty")
    root.mkdir(parents=True,exist_ok=True)
    manifest={"format":"creditratingsystem-u32le-v1","authoring_method":"SHA256 counter bytes v1; no outcome reselection", "experiment_id":experiment_id,"streams":{}}
    for stream,(rows,width) in shapes.items():
        path=root/(stream+".u32le"); checksum=hashlib.sha256()
        with path.open("wb") as f:
            for row in range(rows):
                payload=b"".join(hashlib.sha256(f"creditratingsystem/{experiment_id}/{stream}/{row}/{block}".encode("ascii")).digest() for block in range((width*4+31)//32))[:width*4]
                f.write(payload); checksum.update(payload)
        manifest["streams"][stream]={"file":path.name,"shape":[rows,width],"sha256":checksum.hexdigest()}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory");parser.add_argument("tier",choices=("tiny","standard","stress"));parser.add_argument("--retail",action="store_true");parser.add_argument("--experiment-id",default="reference-v1")
    args=parser.parse_args();result=write_innovations(args.directory,args.tier,args.retail,args.experiment_id)
    print(json.dumps({"streams":len(result["streams"]),"actual_bytes":sum(x["shape"][0]*x["shape"][1]*4 for x in result["streams"].values())}))
