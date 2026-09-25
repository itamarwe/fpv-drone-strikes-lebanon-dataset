#!/usr/bin/env python3
"""Post-fit audit/visualization. Reference pin is evaluation-only and explicit."""
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyproj import Geod,Transformer
from joint_building_constellation import BASE,Problem
from match_terrain_skyline import HgtTile,EARTH_RADIUS_M

def terrain_silhouette(pr,p):
    f,alpha,C,R=pr.unpack(p);C+=pr.origin
    fov=np.degrees(2*np.arctan(640/f));az=np.radians(np.linspace(p[2]-fov*.65,p[2]+fov*.65,641))
    ranges=np.geomspace(450,25000,500)
    east=C[0]+np.sin(az[:,None])*ranges;north=C[1]+np.cos(az[:,None])*ranges
    tr=Transformer.from_crs(2154,4326,always_xy=True);lon,lat=tr.transform(east,north)
    z=HgtTile(BASE/'dem/N43E006.hgt').sample(lat,lon)
    angle=np.arctan2(z-C[2]-ranges**2/(2*EARTH_RADIUS_M),ranges)
    index=np.argmax(angle,axis=1);elev=angle[np.arange(len(az)),index];depth=ranges[index]
    rays=np.c_[np.sin(az)*np.cos(elev),np.cos(az)*np.cos(elev),np.sin(elev)]@R.T
    uv=f*rays[:,:2]/rays[:,2,None]+[640,480]
    return uv,depth

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference-lat',type=float,required=True)
    ap.add_argument('--reference-lon',type=float,required=True);args=ap.parse_args()
    root=BASE/'joint_constellation';modes=['fixed','joint','joint_offset']
    data={m:json.loads((root/m/'refined.json').read_text()) for m in modes}
    frozen={m:hashlib.sha256((root/m/'refined.json').read_bytes()).hexdigest() for m in modes}
    (root/'frozen_predictions.json').write_text(json.dumps(dict(primary='joint',policy='Literal roof centres, jointly estimated focal length; selected before reference evaluation',sha256=frozen),indent=2))
    geod=Geod(ellps='WGS84');evaluation=[]
    for mode in modes:
        r=data[mode]['hypotheses'][0];initial=json.loads((root/mode/'results.json').read_text())
        err=geod.inv(args.reference_lon,args.reference_lat,r['camera_lon'],r['camera_lat'])[2]
        evaluation.append(dict(mode=mode,error_m=err,camera_lat=r['camera_lat'],camera_lon=r['camera_lon'],
            focal_px=r['focal_px'],train_cost=r['train_cost'],check_cost=r['check_cost'],train_matches=r['train_matches'],
            check_matches=r['check_matches'],overall_matches=len(r['pairs']),global_seconds=initial['seconds'],
            refinement_seconds=data[mode]['seconds'],height_fraction=r['roof_to_facade_height_fraction']))
    pr=Problem('joint');r=data['joint']['hypotheses'][0];p=r['parameters'];photo=cv2.cvtColor(cv2.imread(str(BASE/'metadata_free/photo_1.jpg')),cv2.COLOR_BGR2RGB)
    # Fixed-pose shuffled controls are descriptive; not a search-adjusted p-value.
    rng=np.random.default_rng(119);original=pr.uv.copy();controls=[]
    for _ in range(100):
        pr.uv[:,0]=original[rng.permutation(len(original)),0]
        controls.append(pr.assignment(p,pr.check,{q['map_index'] for q in r['train_pairs']})[0])
    pr.uv=original
    skyline,dist=terrain_silhouette(pr,p);crest=(skyline[:,0]>750)&(skyline[:,0]<1020)
    peak=skyline[np.flatnonzero(crest)[np.argmin(skyline[crest,1])]]
    alternatives=[]
    for mode in modes:
        for rank,q in enumerate(data[mode]['hypotheses'],1):
            alternatives.append(dict(mode=mode,training_rank=rank,train_cost=q['train_cost'],check_cost=q['check_cost'],
                error_m=geod.inv(args.reference_lon,args.reference_lat,q['camera_lon'],q['camera_lat'])[2],focal_px=q['focal_px']))
    audit=dict(reference_lat=args.reference_lat,reference_lon=args.reference_lon,reference_use='post-fit evaluation only',
        primary_mode='joint',results=evaluation,
        all_refined_hypotheses=alternatives,
        descriptive_null=dict(type='100 shuffled x-coordinate controls at fixed fitted pose, not reoptimized; not a false-positive rate',
            heldout_cost_median=float(np.median(controls)),heldout_cost_min=float(min(controls)),real_heldout_cost=r['check_cost']),
        independent_silhouette=dict(source='SRTM ~30 m, not used in constellation fitting',peak_in_image_region_xy=peak.tolist(),
            observed_crest_region_xyxy=[899,182,924,191],caveat='Canopy/DEM/datum uncertainty; no correction or pose fitting applied'),
        caveat='One previously disclosed development scene. Not a validated accuracy guarantee; individual false matches remain.')
    (root/'evaluation.json').write_text(json.dumps(audit,indent=2))
    ortho=cv2.cvtColor(cv2.imread(str(BASE/'pnp_reference/ign_ortho.jpg')),cv2.COLOR_BGR2RGB)
    meta=json.loads((BASE/'pnp_reference/ign_ortho.json').read_text());xmin,_,_,ymax=meta['bounds']
    features=json.loads((BASE/'pnp_reference/ign_buildings.geojson').read_text())['features']
    fig,axes=plt.subplots(1,2,figsize=(18,8),layout='constrained');ax=axes[0]
    ax.imshow(photo);ax.set_xlim(0,1280);ax.set_ylim(610,155)
    train_ids={v['observed_id'] for v in r['train_pairs']};check_ids={v['observed_id'] for v in r['check_pairs']}
    assigned={q['observed_id'] for q in r['pairs']}
    for q in r['pairs']:
        a=np.array(q['observed_xy']);b=np.array(q['projected_xy']);color='#03e4fc' if q['observed_id'] in check_ids else '#ffc72b'
        ax.plot(*a,'o',ms=5,mfc='none',mec=color,mew=1.2);ax.plot(*b,'+',ms=6,color=color)
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=.7)
    for q in pr.observed:
        if q['id'] not in assigned:ax.plot(*q['centroid_xy'],'x',ms=7,color='#ff5555')
    ax.plot(skyline[:,0],skyline[:,1],color='#ec38dc',lw=1.3,label='Terrain-only skyline (not fitted)')
    ax.set_title('Photo: detected centres ○, projected centres +; red × unmatched')
    ax.set_xlabel('Image x (px)');ax.set_ylabel('Image y (px)');ax.legend(loc='lower left',fontsize=9)
    ax=axes[1];ax.imshow(ortho);mapxy=[]
    for q in r['pairs']:
        v=np.array(features[q['map_index']]['geometry']['coordinates'][0][0]);xy=np.c_[v[:,0]-xmin-.5,ymax-v[:,1]-.5]
        color='#03e4fc' if q['observed_id'] in check_ids else '#ffc72b';ax.plot(xy[:,0],xy[:,1],color=color,lw=1.3);mapxy.append(xy.mean(axis=0))
    mapxy=np.array(mapxy);ax.set_xlim(max(0,mapxy[:,0].min()-100),min(2800,mapxy[:,0].max()+100))
    ax.set_ylim(min(2800,mapxy[:,1].max()+100),max(0,mapxy[:,1].min()-100))
    ax.set_title('IGN orthophoto: proposed corresponding map buildings')
    ax.set_xlabel('Orthophoto x (1 m pixels)');ax.set_ylabel('Orthophoto y (1 m pixels)')
    err=evaluation[1]['error_m'];fig.suptitle(f'Automatic centre constellation: {err:.1f} m from reference pin | focal {r["focal_px"]:.0f} px\nYellow: training proposals · cyan: held-out proposals · identities require review',fontsize=17)
    fig.savefig(root/'constellation_alignment.png',dpi=160);plt.close(fig)
    # Wider projected roof outlines, without interpreting them as exact mask matches.
    fig,ax=plt.subplots(figsize=(16,8),layout='constrained');ax.imshow(photo);ax.set_xlim(0,1280);ax.set_ylim(610,155)
    f,alpha,C,R=pr.unpack(p)
    for q in r['pairs']:
        v=np.array(features[q['map_index']]['geometry']['coordinates'][0][0])-pr.origin
        cam=(v-C)@R.T;xy=f*cam[:,:2]/cam[:,2,None]+[640,480]
        ax.plot(xy[:,0],xy[:,1],color='#ffce20',lw=.8)
    ax.plot(skyline[:,0],skyline[:,1],color='#ec38dc',lw=1.5)
    ax.set_title('Roof outlines for automatically proposed matches; magenta = independent terrain skyline')
    ax.set_xlabel('Image x (px)');ax.set_ylabel('Image y (px)')
    fig.savefig(root/'projected_roofs_and_skyline.png',dpi=160);plt.close(fig)
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
