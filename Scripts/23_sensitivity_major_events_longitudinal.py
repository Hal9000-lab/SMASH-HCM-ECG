#!/usr/bin/env python3
"""
23_sensitivity_major_events_longitudinal.py

SMASH-HCM -- Step 23
Sensitivity analysis of the Step-17 longitudinal trajectories to major clinical
events/interventions (SRT, atrial fibrillation, pacemaker implantation).

Default inputs
--------------
Data/SHaReData-FLO(3).xlsx
Results/18_Genotype_Longitudinal/genotype_subject_cohort.csv
Results/16_Longitudinal_DualManifold/longitudinal_dual_manifold.csv

Outputs
-------
Results/23_MajorEvents_Longitudinal_Sensitivity/
    trajectory_sensitivity_event_censoring.csv
    cohort_sensitivity_counts.csv
    event_prevalence_exact_longitudinal.csv

Raw/source data remain under Data/. All derived outputs are written to the
numbered Results directory for this step.
"""

import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import csv, zipfile, xml.etree.ElementTree as ET, re, math, warnings, json
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import statsmodels.api as sm

XLSX=Path('Data/SHaReData-FLO.xlsx')
CROSS=Path('Results/18_Genotype_Longitudinal/genotype_subject_cohort.csv')
LONG=Path('Results/16_Longitudinal_DualManifold/longitudinal_dual_manifold.csv')
OUT=Path('Results/23_MajorEvents_Longitudinal_Sensitivity')
OUT.mkdir(parents=True, exist_ok=True)
OUTCOMES=['H_z_01','H_z_05','H_z_06','d_N']
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
RNS='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
EVENTS={
 'myectomy':('Myectomy','Myectomy_Age'),
 'etoh':('EtOH_Ablation','EtOH_Ablation_Age'),
 'af':('Atrial_Fibrillation','Atrial_Fibrillation_Age'),
 'pacer':('Pacer','Pacer_Age'),
 'icd':('ICD','ICD_Age'),
}

def sf(x):
 try:
  y=float(x); return y if math.isfinite(y) else None
 except:return None

def yes(x):return str(x or '').strip().lower() in {'yes','true','1','y'}
def pdte(x):
 try:return datetime.fromisoformat(str(x).replace('Z',''))
 except:return None

def eligible(rows):
 by=defaultdict(list)
 for r in rows:by[r['s']].append(r['t'])
 return {s for s,t in by.items() if len(t)>=2 and len({round(x,10) for x in t})>=2}
def bh(ps):
 a=np.asarray(ps,float); q=np.full(a.shape,np.nan); m=np.isfinite(a)
 if not m.any():return q
 p=a[m]; o=np.argsort(p); r=p[o]; qq=np.minimum.accumulate((r*len(r)/np.arange(1,len(r)+1))[::-1])[::-1]; qq=np.clip(qq,0,1); tmp=np.empty(len(r)); tmp[o]=qq; q[m]=tmp; return q

def load_cross():
 s2p={};p2s={}
 with open(CROSS,newline='',encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):
   if r.get('mapping_status')!='matched' or not r.get('SHaRe_PID'):continue
   try:p=str(int(float(r['SHaRe_PID'])))
   except:continue
   s=r['analysis_subject_key'].strip();s2p[s]=p;p2s[p]=s
 return s2p,p2s

def cn(ref):
 m=re.match(r'([A-Z]+)',ref);n=0
 for ch in m.group(1):n=n*26+ord(ch)-64
 return n-1

def load_events(p2s):
 wanted={'HCM_Events_PID'}
 for a,b in EVENTS.values():wanted|={a,b}
 with zipfile.ZipFile(XLSX) as z:
  shared=[]
  if 'xl/sharedStrings.xml' in z.namelist():
   rt=ET.fromstring(z.read('xl/sharedStrings.xml'))
   shared=[''.join((t.text or '') for t in si.iter(f'{{{NS}}}t')) for si in rt.findall(f'{{{NS}}}si')]
  wb=ET.fromstring(z.read('xl/workbook.xml')); rel=ET.fromstring(z.read('xl/_rels/workbook.xml.rels')); rm={x.attrib['Id']:x.attrib['Target'] for x in rel}
  target=None
  for s in wb.find(f'{{{NS}}}sheets'):
   if s.attrib['name']=='HCM_Events':target=rm[s.attrib[f'{{{RNS}}}id']];break
  root=ET.fromstring(z.read(target if target.startswith('xl/') else 'xl/'+target)); rows=root.find(f'{{{NS}}}sheetData').findall(f'{{{NS}}}row')
  def cv(c):
   typ=c.attrib.get('t');v=c.find(f'{{{NS}}}v')
   if v is None:return ''
   x=v.text or ''
   if typ=='s':
    try:return shared[int(x)]
    except:return x
   return x
  head={cn(c.attrib['r']):cv(c) for c in rows[0].findall(f'{{{NS}}}c')}
  out={}
  for row in rows[1:]:
   d={}
   for c in row.findall(f'{{{NS}}}c'):
    h=head.get(cn(c.attrib['r']))
    if h in wanted:d[h]=cv(c)
   p=d.get('HCM_Events_PID','').strip()
   if p in p2s:out[p]=d
 return out

def load_long(s2p):
 allr=[];exact=[]
 with open(LONG,newline='',encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):
   if r.get('valid_for_latent_projection','').lower() not in {'true','1'}:continue
   s=r['analysis_subject_key'].strip();t=sf(r.get('time_from_first_valid_ecg_years'));dt=pdte(r.get('ecg_time'));bd=pdte(r.get('birthdate'))
   if t is None or dt is None or bd is None:continue
   vals={o:sf(r.get(o)) for o in OUTCOMES}
   if any(v is None for v in vals.values()):continue
   age=(dt-bd).total_seconds()/(365.2425*86400)
   q={'s':s,'t':t,'age':age,**vals};allr.append(q)
   if s in s2p:exact.append(q)
 return allr,exact

def eventmap(s2p,ev):
 out={}
 for s,p in s2p.items():
  d=ev.get(p,{}) ; rr={}
  for k,(fc,ac) in EVENTS.items():
   pos=yes(d.get(fc,'')); rr[k+'_pos']=pos; rr[k+'_age']=sf(d.get(ac,'')) if pos else None
  aa=[rr[k+'_age'] for k in ['myectomy','etoh'] if rr[k+'_age'] is not None]
  rr['srt_pos']=rr['myectomy_pos'] or rr['etoh_pos'];rr['srt_age']=min(aa) if aa else None
  out[s]=rr
 return out

def scenario(rows,em,keys=None):
 base=eligible(rows)
 if not keys:return [r for r in rows if r['s'] in base],{'n_subjects':len(base),'n_visits':sum(r['s'] in base for r in rows),'censored':0,'missing_excluded':0,'lost':0}
 filtered=[];missing=set();cens=0
 for r in rows:
  if r['s'] not in base:continue
  e=em.get(r['s'],{});ages=[];bad=False
  for k in keys:
   if e.get(k+'_pos',False):
    a=e.get(k+'_age')
    if a is None:bad=True
    else:ages.append(a)
  if bad:missing.add(r['s']);continue
  cut=min(ages) if ages else None
  if cut is not None and r['age']>=cut:cens+=1;continue
  filtered.append(r)
 keep=eligible(filtered);rr=[r for r in filtered if r['s'] in keep]
 return rr,{'n_subjects':len(keep),'n_visits':len(rr),'censored':cens,'missing_excluded':len(missing),'lost':len(base-keep)}

def fit_task(task):
 sc,o,dat=task
 by=defaultdict(list)
 for s,t,y in dat:by[s].append((t,y))
 keep={s for s,g in by.items() if len(g)>=2 and len({round(x[0],10) for x in g})>=2}
 d=[x for x in dat if x[0] in keep]
 y=np.asarray([x[2] for x in d],float);t=np.asarray([x[1] for x in d],float);grp=np.asarray([x[0] for x in d],object)
 X=np.column_stack([np.ones(len(d)),t]);Z=X.copy();res=None;method=None;structure=None; attempts=[]
 for m in (['powell','lbfgs','cg'] if o=='H_z_01' else ['lbfgs','powell','cg']):
  try:
   with warnings.catch_warnings(record=True) as ww:
    warnings.simplefilter('always');rr=sm.MixedLM(y,X,groups=grp,exog_re=Z).fit(reml=True,method=m,maxiter=2000,disp=False)
   attempts.append((m,bool(rr.converged),' | '.join(str(w.message) for w in ww)))
   if rr.converged:res=rr;method=m;structure='random_intercept_slope';break
  except Exception as e:attempts.append((m,False,str(e)))
 if res is None:
  Z=np.ones((len(d),1))
  for m in ['lbfgs','powell']:
   try:
    with warnings.catch_warnings(record=True) as ww:
     warnings.simplefilter('always');rr=sm.MixedLM(y,X,groups=grp,exog_re=Z).fit(reml=True,method=m,maxiter=2000,disp=False)
    attempts.append(('RI-'+m,bool(rr.converged),' | '.join(str(w.message) for w in ww)))
    if rr.converged:res=rr;method=m;structure='random_intercept_only';break
   except Exception as e:attempts.append(('RI-'+m,False,str(e)))
 if res is None:return {'scenario':sc,'outcome':o,'success':False,'n_subjects':len(keep),'n_obs':len(d),'attempts':json.dumps(attempts)}
 fe=np.asarray(res.fe_params,float);se=np.asarray(res.bse_fe,float);pv=np.asarray(res.pvalues,float);ci=np.asarray(res.conf_int(),float)
 return {'scenario':sc,'outcome':o,'success':True,'n_subjects':len(keep),'n_obs':len(d),'model_structure':structure,'optimizer':method,'slope':float(fe[1]),'se':float(se[1]),'ci_low':float(ci[1,0]),'ci_high':float(ci[1,1]),'p':float(pv[1]),'attempts':json.dumps(attempts)}

def main():
 for p in (XLSX, CROSS, LONG):
  if not p.exists():
   raise FileNotFoundError(f'Required input not found: {p}')
 s2p,p2s=load_cross();ev=load_events(p2s);em=eventmap(s2p,ev);allr,exact=load_long(s2p)
 specs=[('original_full',allr,None),('exact_mapped',exact,None),('pre_SRT',exact,['srt']),('pre_AF',exact,['af']),('pre_pacer',exact,['pacer']),('pre_core_clean',exact,['srt','af','pacer'])]
 srows={};meta=[]
 for n,r,k in specs:
  rr,mm=scenario(r,em,k);srows[n]=rr;meta.append({'scenario':n,**mm})
 tasks=[]
 for n in srows:
  rr=srows[n]
  for o in OUTCOMES:tasks.append((n,o,[(r['s'],r['t'],r[o]) for r in rr]))
 results=[]
 with ProcessPoolExecutor(max_workers=6) as ex:
  fut=[ex.submit(fit_task,t) for t in tasks]
  for f in as_completed(fut):results.append(f.result())
 order=[x[0] for x in specs]
 results.sort(key=lambda r:(order.index(r['scenario']),OUTCOMES.index(r['outcome'])))
 for sc in order:
  z=[r for r in results if r['scenario']==sc];q=bh([r.get('p',np.nan) for r in z])
  for r,qv in zip(z,q):r['q']=float(qv) if np.isfinite(qv) else ''
 ref={r['outcome']:r for r in results if r['scenario']=='exact_mapped' and r['success']}
 for r in results:
  if r['success'] and r['outcome'] in ref:
   a=ref[r['outcome']]['slope'];b=r['slope'];r['delta_vs_exact']=b-a;r['ratio_vs_exact']=b/a if a else '';r['direction_preserved']=bool(np.sign(a)==np.sign(b))
 # event crossing counts
 exelig=eligible(exact); counts=[]
 for k in ['srt','myectomy','etoh','af','pacer','icd']:
  pos=known=cross=pre=post=0
  for s in exelig:
   e=em.get(s,{})
   if not e.get(k+'_pos',False):continue
   pos+=1;a=e.get(k+'_age')
   if a is None:continue
   known+=1;g=[r for r in exact if r['s']==s];nb=sum(r['age']<a for r in g);na=sum(r['age']>=a for r in g);pre+=nb;post+=na;cross+=int(nb>0 and na>0)
  counts.append({'event':k,'positive_subjects':pos,'known_age':known,'subjects_pre_post':cross,'pre_visits':pre,'post_visits':post})
 # writes
 with open(OUT/'trajectory_sensitivity_event_censoring.csv','w',newline='',encoding='utf-8') as f:
  fields=['scenario','outcome','success','n_subjects','n_obs','model_structure','optimizer','slope','se','ci_low','ci_high','p','q','delta_vs_exact','ratio_vs_exact','direction_preserved','attempts'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
 with open(OUT/'cohort_sensitivity_counts.csv','w',newline='',encoding='utf-8') as f:
  fields=['scenario','n_subjects','n_visits','censored','missing_excluded','lost'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(meta)
 with open(OUT/'event_prevalence_exact_longitudinal.csv','w',newline='',encoding='utf-8') as f:
  fields=['event','positive_subjects','known_age','subjects_pre_post','pre_visits','post_visits'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(counts)
 print('COHORT')
 for m in meta:print(m)
 print('\nEVENTS')
 for c in counts:print(c)
 print('\nRESULTS')
 for o in OUTCOMES:
  print('\n',o)
  for sc in order:
   r=next(x for x in results if x['scenario']==sc and x['outcome']==o)
   print(sc, 'n',r['n_subjects'],'slope',f"{r.get('slope',float('nan')):+.6f}",'CI',f"[{r.get('ci_low',float('nan')):+.6f},{r.get('ci_high',float('nan')):+.6f}]",'p',f"{r.get('p',float('nan')):.3g}",'q',f"{float(r.get('q',float('nan'))):.3g}",'opt',r.get('optimizer'))
if __name__=='__main__':main()
