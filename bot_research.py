# -*- coding: utf-8 -*-
"""TOP-TY-LE-THANG-CAO-GPT — isolated research engine.
Goal: maximize robust short-term profitability, especially >= +2% net after fees.
This repository is independent; it does not import or modify the old bot repository.
"""
from pathlib import Path
import json, os, urllib.parse, urllib.request
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
FEE=.004
NET_TARGET=.02  # default; research grid overrides per configuration
DEV_END=pd.Timestamp("2026-04-23")
HOLD_START=pd.Timestamp("2026-04-24")

def load_market():
    frames=[]
    for p in DATA.glob("*.csv"):
        try:d=pd.read_csv(p)
        except:continue
        if "time" in d.columns:d=d.rename(columns={"time":"date"})
        if not {"date","open","high","low","close","volume"}.issubset(d.columns):continue
        d["date"]=pd.to_datetime(d.date,errors="coerce").dt.normalize()
        for c in ["open","high","low","close","volume"]:d[c]=pd.to_numeric(d[c],errors="coerce")
        d=d.dropna(subset=["date","open","high","low","close"]).sort_values("date").drop_duplicates("date")
        if len(d)<220:continue
        c,h,l,v=d.close,d.high,d.low,d.volume
        ma20=c.rolling(20).mean(); ma50=c.rolling(50).mean(); ma200=c.rolling(200).mean()
        delta=c.diff(); up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); dn=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
        d["rsi"]=100-100/(1+up/dn.replace(0,np.nan))
        d["volr"]=v/v.rolling(20).mean().replace(0,np.nan)
        tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        d["atrp"]=tr.rolling(14).mean()/c*100
        d["ma20d"]=c/ma20-1; d["ret5"]=c.pct_change(5); d["ret20"]=c.pct_change(20)
        d["trend"]=(c>ma20)&(ma20>ma50)&((ma200.isna())|(c>ma200))
        d["hh20"]=h.rolling(20).max().shift(1); d["break20"]=c/d.hh20-1
        d["avgvol20"]=v.rolling(20).mean(); d["days20"]=(v>0).rolling(20).sum()
        d["avgval20bn"]=(c*v).rolling(20).mean()/1_000_000
        d["symbol"]=p.stem.upper(); frames.append(d)
    if not frames: raise RuntimeError("No market CSV files found in data/. Upload historical OHLCV CSVs first.")
    return pd.concat(frames,ignore_index=True).sort_values(["symbol","date"])

def attach_future(x):
    base=x[(x.avgvol20>10000)&(x.days20>=15)&(x.avgval20bn>=3)&x.trend&
           x.rsi.between(42,75)&x.volr.between(.6,4)&x.atrp.between(.8,7)&
           x.ma20d.between(-.02,.12)&x.ret20.between(-.08,.30)].copy()
    gs={s:g.reset_index(drop=True) for s,g in x.groupby("symbol")}; rows=[]
    for z in base.itertuples(index=False):
        g=gs[z.symbol]; ix=g.index[g.date==z.date]
        if not len(ix):continue
        f=g.iloc[int(ix[0])+1:int(ix[0])+11]
        if len(f)<10:continue
        r=z._asdict(); r["entry"]=float(f.iloc[0].open); r["entry_date"]=f.iloc[0].date
        r["future"]=f[["date","high","low","close"]].to_dict("records"); rows.append(r)
    return pd.DataFrame(rows)

def pmask(d,p):
    return d.rsi.between(p["rlo"],p["rhi"])&(d.volr>=p["vr"])&d.atrp.between(.8,p["ahi"])&(d.ma20d<=p["m20"])&(d.ret5>=p["r5"])&(d.ret20>=p["r20"])&(d.avgval20bn>=p["val"])&(d.break20>=p["br"])

def dedup(d):
    # Keep original DataFrame column names intact. Using itertuples()._asdict()
    # renames columns containing dots (e.g. ret_0.015_5) to positional names.
    d=d.copy(); d["score"]=d.volr.clip(0,3)*20+(70-(d.rsi-56).abs())+d.ret20*100
    d=d.sort_values(["date","score"],ascending=[True,False]); keep=[]; last={}
    for idx,r in d.iterrows():
        if r.symbol in last and (r.entry_date-last[r.symbol]).days<14:continue
        keep.append(idx); last[r.symbol]=r.entry_date
    return d.loc[keep].copy()

def one(r,p):
    buy=r.entry; stop=buy*(1-p["stop"]); target=buy*(1+p.get("target",NET_TARGET)+FEE)
    ret=None; why="TIME"; mae=0.; mfe=0.; fut=r.future[:p["hz"]]
    for b in fut:
        mae=min(mae,(float(b["low"])/buy-1)*100); mfe=max(mfe,(float(b["high"])/buy-1)*100)
        if float(b["low"])<=stop: ret=(stop/buy-1-FEE)*100; why="STOP"; break
        if float(b["high"])>=target: ret=(target/buy-1-FEE)*100; why="TARGET"; break
    if ret is None:ret=(float(fut[-1]["close"])/buy-1-FEE)*100
    return ret,why,mae,mfe

def ev(d,p):
    q=dedup(d[pmask(d,p)])
    if q.empty:return None
    a=[one(r,p) for r in q.itertuples(index=False)]; ret=np.array([x[0] for x in a]); rs=np.array([x[1] for x in a])
    pos=ret[ret>0]; neg=ret[ret<0]; pf=pos.sum()/abs(neg.sum()) if len(neg) and neg.sum()!=0 else 99.
    return {"n":len(ret),"win2":float(np.mean(ret>=2-1e-8)*100),"positive":float(np.mean(ret>0)*100),"avg":float(ret.mean()),"median":float(np.median(ret)),
            "stop":float(np.mean(rs=="STOP")*100),"target":float(np.mean(rs=="TARGET")*100),"pf":float(pf),
            "avgwin":float(pos.mean()) if len(pos) else 0,"avgloss":float(neg.mean()) if len(neg) else 0,
            "mae":float(np.mean([x[2] for x in a])),"mfe":float(np.mean([x[3] for x in a]))}

def grid():
    for rlo in [45,48,50,52]:
     for rhi in [62,66,70]:
      if rhi<=rlo:continue
      for vr in [.8,1,1.2,1.5]:
       for ahi in [3.5,5]:
        for m20 in [.03,.06,.10]:
         for r5 in [-.02,0,.02]:
          for r20 in [-.03,.02,.06]:
           for val in [3,10]:
            for br in [-.04,-.01,.01]:
             for target in [.02,.025,.03,.04]:
              for stop in [.015,.02,.025]:
               for hz in [3,5,7,10]:
                yield dict(rlo=rlo,rhi=rhi,vr=vr,ahi=ahi,m20=m20,r5=r5,r20=r20,val=val,br=br,target=target,stop=stop,hz=hz)

def telegram(msg):
    tok=os.getenv("BOT_TOKEN"); chat=os.getenv("CHAT_ID")
    if not tok or not chat: raise RuntimeError("BOT_TOKEN or CHAT_ID GitHub Secret is missing")
    data=urllib.parse.urlencode({"chat_id":chat,"text":msg}).encode()
    raw=urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage",data=data),timeout=20).read()
    response=json.loads(raw)
    if not response.get("ok"): raise RuntimeError("Telegram API rejected the message")
    return response.get("result",{}).get("message_id")

def fast_metrics(q, p):
    """Evaluate precomputed exits; all results are net of round-trip fees."""
    if q.empty: return None
    suffix=f"{p.get('target',NET_TARGET)}_{p['stop']}_{p['hz']}"
    ret=q["ret_"+suffix].to_numpy(dtype=float)
    why=q["why_"+suffix].to_numpy()
    pos=ret[ret>0]; neg=ret[ret<0]
    pf=pos.sum()/abs(neg.sum()) if len(neg) and neg.sum()!=0 else (99. if len(pos) else 0.)
    return {"n":len(ret), "win2":float(np.mean(ret>=2)*100),
            "positive":float(np.mean(ret>0)*100), "avg":float(ret.mean()),
            "median":float(np.median(ret)), "stop":float(np.mean(why=="STOP")*100),
            "target":float(np.mean(why=="TARGET")*100), "pf":float(pf),
            "avgwin":float(pos.mean()) if len(pos) else 0.,
            "avgloss":float(neg.mean()) if len(neg) else 0.}

def ml_probability_research(dev):
    """Walk-forward probability model. Labels/outcomes use Development only."""
    feats=["rsi","volr","atrp","ma20d","ret5","ret20","break20","avgval20bn"]
    d=dev.dropna(subset=feats).copy()
    p={"target":.025,"stop":.02,"hz":7}
    y=[]
    for r in d.itertuples(index=False):
        ret,_,_,_=one(r,p); y.append(int(ret>=2.0))
    d["y"]=y
    probs=pd.Series(np.nan,index=d.index,dtype=float)
    for test_fold in [1,2]:
        tr=d[d.fold<test_fold]; te=d[d.fold==test_fold]
        if len(tr)<200 or te.empty: continue
        model=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1.0,random_state=42)
        model.fit(tr[feats],tr.y)
        probs.loc[te.index]=model.predict_proba(te[feats])[:,1]
    d["prob"]=probs
    reports=[]
    for th in [.55,.60,.65,.70,.75]:
        q=dedup(d[(d.prob>=th)&d.prob.notna()])
        if q.empty:
            continue
        a=[one(r,p) for r in q.itertuples(index=False)]
        ret=np.array([x[0] for x in a]); rs=np.array([x[1] for x in a])
        pos=ret[ret>0]; neg=ret[ret<0]
        pf=pos.sum()/abs(neg.sum()) if len(neg) and neg.sum()!=0 else (99. if len(pos) else 0.)
        e={"n":len(ret),"win2":float(np.mean(ret>=2-1e-8)*100),
           "positive":float(np.mean(ret>0)*100),"avg":float(ret.mean()),
           "median":float(np.median(ret)),"stop":float(np.mean(rs=="STOP")*100),
           "target":float(np.mean(rs=="TARGET")*100),"pf":float(pf),
           "avgwin":float(pos.mean()) if len(pos) else 0.,
           "avgloss":float(neg.mean()) if len(neg) else 0.}
        reports.append({"threshold":th,"metrics":e,"n":len(q)})
    return {"features":feats,"exit":p,"walk_forward":reports}

def main():
    x=load_market()
    c=attach_future(x)
    if c.empty: raise RuntimeError("No complete 10-day future windows in data")
    # Purge Development trades that would observe prices after the Development cutoff.
    # This prevents leakage of Holdout prices into training/selection.
    dev=c[(c.date<=DEV_END)&c.future.map(lambda f: pd.Timestamp(f[9]["date"])<=DEV_END)].copy()
    hold=c[(c.date>=HOLD_START)].copy()
    if dev.empty: raise RuntimeError("Development set empty after purging overlapping trades")
    # Rolling chronological folds derived from available Development dates, not hard-coded years.
    dates=np.sort(dev.date.unique())
    if len(dates)<90: raise RuntimeError("Too few Development dates for three temporal folds")
    cut1=pd.Timestamp(dates[len(dates)//3])
    cut2=pd.Timestamp(dates[2*len(dates)//3])
    dev["fold"]=np.select([dev.date<cut1,dev.date<cut2],[0,1],default=2)
    folds=[{"fold":i,"start":str(dev.loc[dev.fold==i,"date"].min().date()),
            "end":str(dev.loc[dev.fold==i,"date"].max().date())} for i in range(3)]
    # Compute the same exit for each row just eight times total, not on every grid trial.
    for target in [.02,.025,.03,.04]:
        for stop in [.015,.02,.025]:
            for hz in [3,5,7,10]:
                p={"target":target,"stop":stop,"hz":hz}
                outcomes=[one(r,p) for r in dev.itertuples(index=False)]
                suffix=f"{target}_{stop}_{hz}"
                dev["ret_"+suffix]=[z[0] for z in outcomes]
                dev["why_"+suffix]=[z[1] for z in outcomes]
    candidates=[]; near=[]; tested=0; screened=0
    cached_key=None; cached_q=None
    for p in grid():
        tested+=1
        key=tuple((k,v) for k,v in p.items() if k not in ("target","stop","hz"))
        if key!=cached_key:
            cached_q=dedup(dev[pmask(dev,p)])
            cached_key=key
        q=cached_q
        if len(q)<80: continue
        e=fast_metrics(q,p)
        if e["avg"]<=0 or e["pf"]<1.10: continue
        screened+=1
        fs=[fast_metrics(q[q.fold==i],p) for i in range(3)]
        fold_ok=all(z and z["n"]>=15 and z["avg"]>0 and z["pf"]>=1.10 for z in fs)
        score=e["win2"]*1.35+e["positive"]*.25+e["avg"]*12+min(e["pf"],3)*5-e["stop"]*.55+e["avgloss"]*2
        diag={"score":float(score),"params":p,"development":e,"folds":fs,"robust":bool(fold_ok)}
        near.append(diag)
        if fold_ok:
            floor=min(z["avg"] for z in fs)
            dispersion=float(np.std([z["avg"] for z in fs]))
            candidates.append((float(score+floor*10-min(dispersion,5)*2),diag))
    near.sort(key=lambda z:z["score"],reverse=True)
    candidates.sort(key=lambda z:z[0],reverse=True)
    ml_diag=ml_probability_research(dev)
    diagnostics={"ml_probability_research":ml_diag,"tested":tested,"screened":screened,"robust_count":len(candidates),
                 "development_rows":len(dev),"holdout_rows_not_used_for_tuning":len(hold),
                 "fold_boundaries":folds,"top_development_candidates":near[:30],
                 "top_robust_candidates":[z for _,z in candidates[:15]],
                 "holdout_not_accessed_for_tuning":True}
    Path("development_diagnostics.json").write_text(
        json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding="utf-8")
    if not candidates:
        print("NO_ROBUST_CONFIGURATION",json.dumps({"tested":tested,"screened":screened,
              "folds":folds,"diagnostics":"development_diagnostics.json"},ensure_ascii=False),flush=True)
        raise RuntimeError("No robust configuration passed the three chronological Development folds; Holdout remains unopened")
    winner=candidates[0][1]
    p=winner["params"]
    de=winner["development"]
    fs=winner["folds"]
    # LOCK: no parameter adjustment after reading the Holdout.
    ho=ev(hold,p)
    result={"tested":tested,"screened":screened,"robust_passed":len(candidates),
            "locked_params":p,"development":de,"development_folds":fs,
            "holdout":ho,"fee_pct":FEE*100,"target_net_pct":p.get("target",NET_TARGET)*100,
            "data_max":str(x.date.max().date()),"fold_boundaries":folds}
    Path("result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("LOCKED_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
    h=ho or {}
    message=("🏆 TOP-TY-LE-THANG-CAO-GPT — KẾT QUẢ KHÓA" + chr(10)
      + f"DEV: {de['n']} lệnh | ≥2%: {de['win2']:.1f}% | Có lãi: {de['positive']:.1f}% | TB: {de['avg']:+.2f}% | Stop: {de['stop']:.1f}% | PF: {de['pf']:.2f}" + chr(10)
      + f"HOLDOUT: {h.get('n',0)} lệnh | ≥2%: {h.get('win2',float('nan')):.1f}% | Có lãi: {h.get('positive',float('nan')):.1f}% | TB: {h.get('avg',float('nan')):+.2f}% | Stop: {h.get('stop',float('nan')):.1f}% | PF: {h.get('pf',float('nan')):.2f}" + chr(10)
      + f"Grid {tested:,} | Robust {len(candidates)} | phí 0.4%")
    msg_id=telegram(message)
    result["telegram_message_id"]=msg_id
    result["telegram_delivered"]=bool(msg_id)
    Path("result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
