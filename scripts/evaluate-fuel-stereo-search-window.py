#!/usr/bin/env python3
"""Test offset stereo disparity windows on synchronized saved FUEL recordings."""
import argparse,json,math,os,statistics,time
from pathlib import Path
import cv2
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
D=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--configs',default='0:128,0:64,4:64,8:64',help='comma-separated minDisparity:numDisparities pairs')
parser.add_argument('--output',default='media/frc-fuel-stereo-search-offset-sweep.json')
args=parser.parse_args()
CONFIGS=tuple(tuple(map(int,value.split(':'))) for value in args.configs.split(','));STRIDE=20;TARGETS=((808,589),(902,591))
CAL='media/calibration/stereo_calibration_live_order_candidate.yml'
os.environ.update(SCOUT_STEREO_CALIBRATION=CAL,SCOUT_STEREO_RECTIFY='1',SCOUT_STEREO_BASELINE_M='.0635',SCOUT_STEREO_DISPARITY_SCALE='1',SCOUT_STEREO_MATCH_SCALE='.5',SCOUT_STEREO_NUM_DISPARITIES='128',SCOUT_STEREO_VERTICAL_ROI_PAD='24')
from scout.detectors import Detection
from scout.stereo import StereoDepth

def med(x):return statistics.median(x) if x else None
def mad(x):
 m=med(x);return med([abs(v-m) for v in x]) if x else None
def summary(x):return {'n':len(x),'median_m':round(med(x),4),'mad_m':round(mad(x),4),'p05_m':round(sorted(x)[int(.05*(len(x)-1))],4),'p95_m':round(sorted(x)[int(.95*(len(x)-1))],4)} if x else None

def matched(row):
 out=[None,None]
 for b in sorted((x for x in row['detections'] if x['confidence']>=.05),key=lambda b:b['confidence'],reverse=True):
  cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2;dist,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
  if dist<=60 and out[i] is None:out[i]=Detection(int(b['x1']),int(b['y1']),int(b['x2']),int(b['y2']),0,float(b['confidence']))
 return out

out_path=ROOT/args.output
stereo=StereoDepth();out={'method':f"On every twentieth synchronized frame from both saved videos, reuse confidence-0.05 FRC9550 boxes and active rectification/vertical ROI. Compare SGBM 3WAY search windows (minDisparity,numDisparities) {CONFIGS}, with block 5, uniqueness 8, and otherwise fixed settings.",'limitations':'Only stationary FUEL balls around 1.4-1.6 m; no measured depth truth or close/far field cases. Offset windows deliberately cap the valid range interval; the computed coverage limits do not account for calibration error or matcher invalidation at textureless pixels.','calibration':CAL,'baseline_m':.0635,'match_scale':.5,'sample_stride':STRIDE,'configurations':[],'results':[]}
for min_d,num_d in CONFIGS:
 stereo.matcher=cv2.StereoSGBM_create(minDisparity=min_d,numDisparities=num_d,blockSize=5,P1=200,P2=800,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
 started=time.perf_counter();config={'min_disparity_half_px':min_d,'num_disparities_half_px':num_d,'results':[]}
 for clip in D['results']:
  cap=cv2.VideoCapture(str(ROOT/clip['video']));ranges=[[],[]];spacing=[];times=[];processed=0
  for idx,row in enumerate(clip['per_frame']):
   if idx%STRIDE:continue
   # Seek explicitly to sampled frame; this avoids decoding every intervening frame.
   cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
   if not ok:continue
   boxes=matched(row);t0=time.perf_counter();stereo.estimate(frame[:,:1600],frame[:,1600:],[b for b in boxes if b is not None]);times.append((time.perf_counter()-t0)*1000);processed+=1
   for i,b in enumerate(boxes):
    if b is not None and b.distance_m is not None:ranges[i].append(float(b.distance_m))
   if all(b is not None and b.distance_m is not None for b in boxes):
    z=[float(b.distance_m) for b in boxes]
    # Lateral centers use the same rectified projection for every matcher.
    _,_,_,kl,r1,p1=stereo.maps;pts=[]
    for b in boxes:
     c=np.array([[[float(b.x1+b.x2)/2,float(b.y1+b.y2)/2]]],np.float32)
     uv=cv2.undistortPoints(c,kl,stereo.calibration['left_distortion'],R=r1,P=p1)[0,0]
     pts.append(((uv[0]-p1[0,2])*b.distance_m/p1[0,0],(uv[1]-p1[1,2])*b.distance_m/p1[1,1],b.distance_m))
    spacing.append(math.dist(*pts))
  cap.release();focal=stereo.focal_px;baseline=stereo.baseline_m
  max_full=(min_d+num_d-1)/.5;min_range=baseline*focal/max_full
  max_range=(baseline*focal/(min_d/.5)) if min_d else None
  config['results'].append({'scene':clip['scene'],'sampled_frames':processed,'valid_ranges':[len(x) for x in ranges],'range_by_target':[summary(x) for x in ranges],'pair_spacing':summary(spacing),'matcher_ms_median':round(med(times),2) if times else None,'matcher_ms_p95':round(sorted(times)[int(.95*(len(times)-1))],2) if times else None,'theoretical_range_coverage_m':{'near_minimum_m':round(min_range,3),'far_maximum_m':round(max_range,3) if max_range else None}})
  print('window',min_d,num_d,config['results'][-1])
 out['configurations'].append({'min_disparity_half_px':min_d,'num_disparities_half_px':num_d})
 out['results'].append(config)
 out_path.write_text(json.dumps(out,indent=2)+'\n')
 print('config elapsed',round(time.perf_counter()-started,1),'s')
print('saved',out_path)
