#!/usr/bin/env python3
"""Pixel-only semantic point-set registration against metric IGN roof centres.

No building identities, manual corners or reference pin enter search. World is
Lambert-93 EN/up, camera rows are right/down/forward. Xc=R(Xw-C), perspective
u=f*Xc/Zc+cx. Camera elevation is interpolated terrain + 2 m (explicit prior).
Bounding-box centres are noisy semantic proxies, not physical keypoints.
"""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution,linear_sum_assignment,minimize
from scipy.spatial import cKDTree
from pyproj import Transformer

BASE=Path(__file__).resolve().parents[2]/'benchmarks/france_phone_geolocation'

def rotation(yaw,pitch,roll):
    yaw,pitch,roll=np.radians([yaw,pitch,roll])
    forward=np.array([np.sin(yaw)*np.cos(pitch),np.cos(yaw)*np.cos(pitch),np.sin(pitch)])
    right=np.array([np.cos(yaw),-np.sin(yaw),0.]);down=np.cross(forward,right)
    return np.array([right*np.cos(roll)+down*np.sin(roll),down*np.cos(roll)-right*np.sin(roll),forward])

class Problem:
    def __init__(self,mode='fixed',seed=7):
        self.mode=mode;self.focal=2084.352670328932;self.calls=0
        raw=json.loads((BASE/'sam_photo1/segments.json').read_text())['ground']['features']
        selected=[]
        for p in sorted(raw,key=lambda a:-a['score']):
            x0,y0,x1,y1=p['bbox_xyxy'];w=x1-x0;h=y1-y0
            if not (p['score']>=.35 and 100<=p['area_px']<=2500 and 8<w<100 and x0>5 and x1<1275 and 180<y0<570):continue
            if any(np.linalg.norm(np.array(p['centroid_xy'])-np.array(q['centroid_xy']))<max(5,.2*min(w,q['bbox_xyxy'][2]-q['bbox_xyxy'][0])) for q in selected):continue
            selected.append(p)
        self.observed=selected;self.uv=np.array([p['centroid_xy'] for p in selected])
        if len(selected)<12:raise ValueError('At least 12 usable building detections are required')
        self.wh=np.array([[p['bbox_xyxy'][2]-p['bbox_xyxy'][0],p['bbox_xyxy'][3]-p['bbox_xyxy'][1]] for p in selected])
        rng=np.random.default_rng(seed);order=rng.permutation(len(selected))
        self.train=np.sort(order[:round(.7*len(order))]);self.check=np.sort(order[round(.7*len(order)):])
        features=json.loads((BASE/'pnp_reference/ign_buildings.geojson').read_text())['features']
        self.map=[];xyz=[];axes=[];heights=[]
        for index,item in enumerate(features):
            v=np.array(item['geometry']['coordinates'][0][0],float)
            local=(v[:,:2]-v[0,:2]).astype(np.float32);m=cv2.moments(local)
            if not 45<=m['m00']<=2500 or not np.isfinite(v).all():continue
            c=np.array([m['m10'],m['m01']])/m['m00']+v[0,:2]
            box=cv2.boxPoints(cv2.minAreaRect(local));a=(box[1]-box[0])/2;b=(box[2]-box[1])/2
            z=float(np.median(v[:,2]));height=float(item['properties'].get('hauteur') or 5)
            if not -10<z<500 or not 0<height<60:continue
            xyz.append([*c,z]);axes.append([a,b]);heights.append(height)
            self.map.append(dict(index=index,id=item['properties']['cleabs'],area_m2=m['m00']))
        self.xyz=np.array(xyz);self.axes=np.array(axes);self.heights=np.array(heights)
        t=np.load(BASE/'pnp_reference/camera_terrain.npz')
        self.terrain=RegularGridInterpolator((t['y'],t['x']),t['z'],bounds_error=False,fill_value=np.nan)
        self.origin=np.array([996000.,6254000.,0.]);self.xyz-=self.origin
        self.bounds=[(t['x'][0]-self.origin[0]+1,t['x'][-1]-self.origin[0]-1),
                     (t['y'][0]-self.origin[1]+1,t['y'][-1]-self.origin[1]-1),(-20,35),(-18,5),(-5,5)]
        if mode!='fixed':self.bounds.append((np.log(1300),np.log(3500)))
        if mode=='joint_offset':self.bounds.append((0,.75))
        self.obs_feat=np.c_[self.uv/[8,6],np.log(self.wh[:,0])/.65]

    def unpack(self,p):
        f=self.focal if self.mode=='fixed' else np.exp(p[5])
        alpha=p[6] if self.mode=='joint_offset' else 0.
        c=np.array([p[0],p[1],float(self.terrain([[p[1]+self.origin[1],p[0]+self.origin[0]]])[0])+2])
        return f,alpha,c,rotation(*p[2:5])

    def project(self,p):
        f,alpha,c,R=self.unpack(p);xyz=self.xyz.copy();xyz[:,2]-=alpha*self.heights
        v=(xyz-c)@R.T;z=v[:,2];safe=np.maximum(z,1)
        uv=f*v[:,:2]/safe[:,None]+[640,480]
        widths=2*f*np.sum(np.abs(self.axes@R[0,:2]),axis=1)/safe
        keep=(z>60)&(z<5000)&(uv[:,0]>-60)&(uv[:,0]<1340)&(uv[:,1]>150)&(uv[:,1]<620)&(widths>6)&(widths<180)
        ids=np.where(keep)[0];return uv[keep],widths[keep],ids

    def coarse(self,p):
        self.calls+=1;uv,w,ids=self.project(p)
        if len(ids)<15:return 20.
        feat=np.c_[uv/[8,6],np.log(w)/.65]
        d,i=cKDTree(feat).query(self.obs_feat[self.train],k=1)
        d2=np.minimum(d*d,9);unique=len(np.unique(i[d<3]))
        return float(d2.mean()+.6*(1-unique/len(i))+.08*np.log(max(len(ids),1)/150))

    def assignment(self,p,subset,excluded_map_indices=None):
        if len(subset)==0:raise ValueError('Assignment requires a nonempty observation subset')
        uv,w,ids=self.project(p)
        if excluded_map_indices:
            keep=np.array([self.map[i]['index'] not in excluded_map_indices for i in ids],dtype=bool)
            uv,w,ids=uv[keep],w[keep],ids[keep]
        if len(ids)==0:return 9.,[]
        observed=self.uv[subset];wh=self.wh[subset]
        sigma=np.maximum([4,4],wh*[.16,.30])
        delta=(observed[:,None]-uv[None])/sigma[:,None]
        cost=np.sum(delta*delta,axis=2)+.35*(np.log(wh[:,0,None]/w[None])/.65)**2
        # Dedicated dummy assignments allow all detections to remain unmatched.
        aug=np.c_[cost,np.full((len(subset),len(subset)),9.)]
        row,col=linear_sum_assignment(aug);values=aug[row,col];pairs=[]
        for a,b,value in zip(row,col,values):
            if b>=len(ids) or value>=9:continue
            oi=int(subset[a]);mi=int(ids[b])
            pairs.append(dict(observed_index=oi,observed_id=self.observed[oi]['id'],map_index=self.map[mi]['index'],
                map_id=self.map[mi]['id'],observed_xy=self.uv[oi].tolist(),projected_xy=uv[b].tolist(),
                observed_width_px=float(wh[a,0]),projected_width_px=float(w[b]),cost=float(value),
                pixel_distance=float(np.linalg.norm(self.uv[oi]-uv[b]))))
        return float(values.mean()),pairs

    def payload(self,p):
        f,alpha,c,R=self.unpack(p);C=c+self.origin
        lon,lat=Transformer.from_crs(2154,4326,always_xy=True).transform(*C[:2])
        train,tp=self.assignment(p,self.train)
        check,cp=self.assignment(p,self.check,{q['map_index'] for q in tp})
        # Overall one-to-one assignment is separate from held-out diagnostics.
        overall,allpairs=self.assignment(p,np.arange(len(self.uv)))
        return dict(parameters=np.asarray(p).tolist(),focal_px=float(f),roof_to_facade_height_fraction=float(alpha),
            camera_lambert93_xyz=C.tolist(),camera_lon=lon,camera_lat=lat,yaw_pitch_roll_deg=np.asarray(p[2:5]).tolist(),
            train_cost=train,train_matches=len(tp),check_cost=check,check_matches=len(cp),
            check_assignment_policy='Map instances assigned to training detections cannot be reused by held-out detections',
            overall_cost=overall,pairs=allpairs,train_pairs=tp,check_pairs=cp,coarse_cost=self.coarse(p))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['fixed','joint','joint_offset'],default='fixed')
    ap.add_argument('--iterations',type=int,default=240);ap.add_argument('--seeds',default='7,19,41')
    args=ap.parse_args();out=BASE/'joint_constellation'/args.mode;out.mkdir(parents=True,exist_ok=True)
    pr=Problem(args.mode);start=time.perf_counter();pool=[];runs=[]
    print('mode',args.mode,'observed',len(pr.observed),'train',len(pr.train),'check',len(pr.check),'map',len(pr.map),flush=True)
    for seed in map(int,args.seeds.split(',')):
        t=time.perf_counter()
        fit=differential_evolution(pr.coarse,pr.bounds,seed=seed,popsize=16,maxiter=args.iterations,
            tol=.0002,polish=False,updating='immediate')
        order=np.argsort(fit.population_energies)[:8]
        pool.extend(fit.population[order].tolist())
        runs.append(dict(seed=seed,seconds=time.perf_counter()-t,cost=float(fit.fun),evaluations=int(fit.nfev)))
        print('global',runs[-1],flush=True)
    # Deduplicate coarse states before one-to-one refinement; no check-point scores enter selection.
    unique=[]
    for p in sorted(pool,key=pr.coarse):
        if all(np.linalg.norm(np.array(p[:2])-np.array(q[:2]))>35 or abs(p[2]-q[2])>1 for q in unique):unique.append(p)
    results=[]
    for p in unique[:12]:
        fit=minimize(lambda q:pr.assignment(q,pr.train)[0],p,method='Powell',bounds=pr.bounds,
            options=dict(maxiter=45,xtol=1e-4,ftol=1e-5))
        # Powell can escape a narrow good basin while line-searching; preserve the original.
        best=fit.x if pr.assignment(fit.x,pr.train)[0]<pr.assignment(p,pr.train)[0] else p
        results.append(pr.payload(best))
    results.sort(key=lambda r:r['train_cost'])
    if not results:raise RuntimeError('Search produced no usable hypotheses')
    payload=dict(mode=args.mode,seconds=time.perf_counter()-start,global_runs=runs,
        bounds=pr.bounds,parameter_names=['east_offset','north_offset','yaw_deg','pitch_deg','roll_deg']+
            ([] if args.mode=='fixed' else ['log_focal'])+(['height_fraction'] if args.mode=='joint_offset' else []),
        train_indices=pr.train.tolist(),check_indices=pr.check.tolist(),observations=pr.observed,
        map_buildings=len(pr.map),hypotheses=results,
        assumptions=dict(principal_point_px=[640,480],distortion=0,camera_height_above_terrain_m=2,
            terrain_sampling_m=50,occlusion_model='none; unmatched map points allowed',
            manual_correspondences=False,reference_pin_used=False),
        limitations='Semantic-centre proxies; dense-map false matches possible. Held-out detections share one scene and are not independent photographs. No accepted geolocation without external review.')
    (out/'results.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(dict(seconds=payload['seconds'],hypotheses=len(results),top={k:v for k,v in results[0].items() if 'pairs' not in k}),indent=2))

if __name__=='__main__':main()
