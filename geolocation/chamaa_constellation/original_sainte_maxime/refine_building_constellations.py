#!/usr/bin/env python3
"""Cross-seed all three ablations fairly, using geometry costs only, no pin.

Discovery by one model supplies a basin to the other models. This avoids
attributing a different optimizer basin to the facade-height correction alone.
All refinement and selection use training detections only.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from joint_building_constellation import BASE,Problem

def main():
    modes=['fixed','joint','joint_offset'];source=[]
    for mode in modes:
        d=json.loads((BASE/f'joint_constellation/{mode}/results.json').read_text())
        for r in d['hypotheses']:
            source.append(dict(source_mode=mode,parameters=r['parameters'],focal=r['focal_px'],alpha=r['roof_to_facade_height_fraction']))
    for mode in modes:
        p=Problem(mode);start=time.perf_counter();starts=[]
        for row in source:
            q=row['parameters'][:5]
            if mode!='fixed':q=q+[np.log(row['focal'])]
            if mode=='joint_offset':q=q+[row['alpha']]
            starts.append(q)
        starts=sorted(starts,key=lambda q:p.assignment(q,p.train)[0]);results=[]
        for index,q in enumerate(starts[:2]):
            radius=[350,350,5,3,3]+([] if mode=='fixed' else [.20])+([.45] if mode=='joint_offset' else [])
            bounds=[(max(lo,v-r),min(hi,v+r)) for (lo,hi),v,r in zip(p.bounds,q,radius)]
            fit=differential_evolution(lambda x:p.assignment(x,p.train)[0],bounds,seed=71+index,
                popsize=12,maxiter=150,tol=.0001,polish=False,x0=q)
            best=fit.x if p.assignment(fit.x,p.train)[0]<p.assignment(q,p.train)[0] else q
            results.append(p.payload(best))
            print(mode,index,results[-1]['train_cost'],results[-1]['check_cost'],flush=True)
        results.sort(key=lambda r:r['train_cost'])
        # Selection costs are frozen before any reference-location evaluation.
        data=dict(mode=mode,seconds=time.perf_counter()-start,hypotheses=results,
            train_indices=p.train.tolist(),check_indices=p.check.tolist(),observations=p.observed,
            reference_pin_used=False,method='Training-only one-to-one local DE; cross-seeded from all global-model candidates',
            camera_height_above_terrain_m=2,principal_point_px=[640,480],zero_distortion_assumed=True)
        (BASE/f'joint_constellation/{mode}/refined.json').write_text(json.dumps(data,indent=2))
        print(mode,'seconds',data['seconds'],flush=True)

if __name__=='__main__':main()
