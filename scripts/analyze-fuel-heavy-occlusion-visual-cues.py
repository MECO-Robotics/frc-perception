#!/usr/bin/env python3
"""Compare visible shape cues in mild and heavy FUEL occlusion captures."""
import json, math, statistics
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CLIPS={'mild':'media/frc-fuel-live-occlusion-auto-20261103.mp4','heavy':'media/frc-fuel-live-heavy-occlusion-20261103.mp4'}
ROI=(680,500,1010,660)  # left-view crop enclosing the pair
TIMES=[.5*i for i in range(1,60)]
def median(xs): return round(float(statistics.median(xs)),3) if xs else None
results={}
for scene,path in CLIPS.items():
 cap=cv2.VideoCapture(str(ROOT/path));fps=cap.get(cv2.CAP_PROP_FPS);rows=[]
 for t in TIMES:
  cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
  if not ok: continue
  x0,y0,x1,y1=ROI;img=frame[y0:y1,x0:x1];hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  mask=cv2.inRange(hsv,(10,150,70),(40,255,255))
  mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
  mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
  n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
  candidates=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=500]
  largest=None
  if candidates:
   i=max(candidates,key=lambda j:stats[j,cv2.CC_STAT_AREA]);x,y,w,h,area=map(int,stats[i]);comp=(labels[y:y+h,x:x+w]==i).astype(np.uint8)
   contours,_=cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);contour=max(contours,key=cv2.contourArea);a=cv2.contourArea(contour);per=cv2.arcLength(contour,True);hull=cv2.convexHull(contour);ha=cv2.contourArea(hull)
   ellipse=cv2.fitEllipse(contour) if len(contour)>=5 else None;pts=contour.reshape(-1,2).astype(float)
   A=np.c_[2*pts[:,0],2*pts[:,1],np.ones(len(pts))];rhs=(pts**2).sum(axis=1);cx,cy,c=np.linalg.lstsq(A,rhs,rcond=None)[0];radius=math.sqrt(max(0,cx*cx+cy*cy+c));res=np.sqrt(np.mean((np.hypot(pts[:,0]-cx,pts[:,1]-cy)-radius)**2))/radius if radius else 0
   dist=cv2.distanceTransform(comp,cv2.DIST_L2,5);pm=(dist>=max(12,float(dist.max())*.62)).astype(np.uint8);pn,pl=cv2.connectedComponents(pm,8);peaks=[]
   for j in range(1,pn):
    yy,xx=np.where(pl==j)
    if len(xx)>1:peaks.append([round(float(xx.mean()+x+x0),1),round(float(yy.mean()+y+y0),1)])
   largest={'area_px':area,'bbox_xywh_px':[x+x0,y+y0,w,h],'component_count_ge500':len(candidates),'circularity':round(4*math.pi*a/(per*per),3) if per else None,'solidity':round(a/ha,3) if ha else None,'ellipse_minor_major_ratio':round(min(ellipse[1])/max(ellipse[1]),3) if ellipse else None,'circle_fit_radius_px':round(radius,2),'circle_fit_normalized_radial_rmse':round(float(res),3),'distance_transform_peak_count':len(peaks),'distance_transform_peaks_xy':peaks}
  gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);blur=cv2.GaussianBlur(gray,(9,9),2)
  circles=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,dp=1.2,minDist=30,param1=90,param2=25,minRadius=30,maxRadius=65)
  found=[]
  if circles is not None:
   for xx,yy,r in circles[0]:
    ax,ay=float(xx+x0),float(yy+y0)
    if 740<=ax<=960 and 530<=ay<=640:found.append({'center_xy_px':[round(ax,1),round(ay,1)],'radius_px':round(float(r),1)})
  pair=None
  if len(found)>=2:
   pair=min(({'center_gap_px':round(math.dist(a['center_xy_px'],b['center_xy_px']),1),'radius_sum_px':round(a['radius_px']+b['radius_px'],1),'gap_over_radius_sum':round(math.dist(a['center_xy_px'],b['center_xy_px'])/(a['radius_px']+b['radius_px']),3)} for i,a in enumerate(found) for b in found[i+1:]),key=lambda q:q['gap_over_radius_sum'])
  rows.append({'time_s':t,'hsv_mask_area_px':int(mask.sum()/255),'largest_component':largest,'hough_circles_near_pair':found,'closest_hough_pair_geometry':pair})
 cap.release()
 valid=[r for r in rows if r['largest_component']]
 circ=[r['largest_component']['circularity'] for r in valid];ell=[r['largest_component']['ellipse_minor_major_ratio'] for r in valid];raderr=[r['largest_component']['circle_fit_normalized_radial_rmse'] for r in valid];dt=[r['largest_component']['distance_transform_peak_count'] for r in valid]
 hcounts=[len(r['hough_circles_near_pair']) for r in rows];pairs=[r['closest_hough_pair_geometry'] for r in rows if r['closest_hough_pair_geometry']]
 results[scene]={'video':path,'fps':fps,'sample_times_s':TIMES,'samples':len(rows),'roi_xyxy_px':ROI,'hsv_config':{'hue':[10,40],'saturation_min':150,'value_min':70,'morphology':'3x3 open then 5x5 close'},
  'median_hsv_mask_area_px':median([r['hsv_mask_area_px'] for r in rows]),'median_largest_component_area_px':median([r['largest_component']['area_px'] for r in valid]),'largest_component_bounding_box_wh_px_median':[median([r['largest_component']['bbox_xywh_px'][2] for r in valid]),median([r['largest_component']['bbox_xywh_px'][3] for r in valid])],
  'largest_component_circularity_median':median(circ),'largest_component_ellipse_minor_major_ratio_median':median(ell),'circle_fit_normalized_radial_rmse_median':median(raderr),'distance_transform_peak_count_histogram':{str(k):dt.count(k) for k in sorted(set(dt))},
  'hough_param2_25_radius_30_65_hypotheses_near_pair_histogram':{str(k):hcounts.count(k) for k in sorted(set(hcounts))},'hough_two_circle_pair_fraction':round(sum(k>=2 for k in hcounts)/len(hcounts),3) if hcounts else None,'hough_two_circle_center_gap_median_px':median([p['center_gap_px'] for p in pairs]),'hough_radii_sum_median_px':median([p['radius_sum_px'] for p in pairs]),'hough_center_gap_over_radii_sum_median':median([p['gap_over_radius_sum'] for p in pairs]),'per_sample':rows}
 print(scene,{k:v for k,v in results[scene].items() if k!='per_sample'})
out={'method':'Compare 59 time samples from each fixed-view capture. Apply HSV segmentation and contour geometry in a manually selected left-view ROI; run HoughCircles on the grayscale ROI and retain candidate circles near the pair. The mild capture is the previous side-by-side arrangement; the heavy capture is the substantially overlapping arrangement.','interpretation_limits':['This compares two stationary arrangements at roughly the same camera distance and illumination; the thresholds/ROI are exploratory and should not be treated as general-purpose settings.','Hough circle candidates may fit ball texture, seat seams, or partial arcs. Two hypotheses are evidence for visible circular structure, not proof of two balls.','The contour circle-fit residual is measured against one circle fitted to the union mask; it does not directly estimate physical hidden surface area.'],'results':results}
(ROOT/'media/frc-fuel-occlusion-visual-cues-20261103.json').write_text(json.dumps(out,indent=2)+'\n')
print('saved media/frc-fuel-occlusion-visual-cues-20261103.json')
