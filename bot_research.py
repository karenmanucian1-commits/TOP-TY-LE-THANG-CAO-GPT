# -*- coding: utf-8 -*-
"""TOP-TY-LE-THANG-CAO-GPT — isolated research engine.
Goal: maximize robust short-term profitability, especially >= +2% net after fees.
This repository is independent; it does not import or modify the old bot repository.
"""
from pathlib import Path
import json, os, urllib.parse, urllib.request
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
FEE=.004
NET_TARGET=.02
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
        if len(f)<3:continue
        r=z._asdict(); r["entry"]=float(f.iloc[0].open); r["entry_date"]=f.iloc[0].date
        r["future"]=f[["date","high","low","close"]].to_dict("records"); rows.append(r)
    return pd.DataFrame(rows)

def pmask(d,p):
    return d.rsi.between(p["rlo"],p["rhi"])&(d.volr>=p["vr"])&d.atrp.between(.8,p["ahi"])&(d.ma20d<=p["m20"])&(d.ret5>=p["r5"])&(d.ret20>=p["r20"])&(d.avgval20bn>=p["val"])&(d.break20>=p["br"])

def dedup(d):
    d=d.copy(); d["score"]=d.volr.clip(0,3)*20+(70-(d.rsi-56).abs())+d.ret20*100
    d=d.sort_values(["date","score"],ascending=[True,False]); out=[]; last={}
    for r in d.itertuples(index=False):
        if r.symbol in last and (r.entry_date-last[r.symbol]).days<14:continue
        out.append(r._asdict()); last[r.symbol]=r.entry_date
    return pd.DataFrame(out)

def one(r,p):
    buy=r.entry; stop=buy*(1-p["stop"]); target=buy*(1+NET_TARGET+FEE)
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
    return {"n":len(ret),"win2":float(np.mean(ret>=2)*100),"positive":float(np.mean(ret>0)*100),"avg":float(ret.mean()),"median":float(np.median(ret)),
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
             for stop in [.015,.02,.025,.03]:
              for hz in [5,10]:
               yield dict(rlo=rlo,rhi=rhi,vr=vr,ahi=ahi,m20=m20,r5=r5,r20=r20,val=val,br=br,stop=stop,hz=hz)

def telegram(msg):
    tok=os.getenv("BOT_TOKEN"); chat=os.getenv("CHAT_ID")
    if not tok or not chat:return
    data=urllib.parse.urlencode({"chat_id":chat,"text":msg}).encode()
    urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage",data=data),timeout=20).read()

def main():
    x=load_market(); c=attach_future(x)
    dev=c[c.date<=DEV_END].copy(); hold=c[c.date>=HOLD_START].copy()
    folds=[dev[(dev.date>=a)&(dev.date<=b)] for a,b in [
      (pd.Timestamp("2023-01-01"),pd.Timestamp("2024-01-31")),
      (pd.Timestamp("2024-02-01"),pd.Timestamp("2025-03-31")),
      (pd.Timestamp("2025-04-01"),DEV_END)]]
    # Stage 1: screen all configurations on Development. Keep broad top candidates,
    # then demand positive expectancy in every Development fold.
    shortlist=[]; tested=0
    for p in grid():
        tested+=1; e=ev(dev,p)
        if not e or e["n"]<80 or e["avg"]<=0 or e["pf"]<=1: continue
        # Risk-adjusted objective: >=2% wins matter most, but stop rate/loss severity are penalized.
        score=e["win2"]*1.35+e["positive"]*.25+e["avg"]*12+min(e["pf"],3)*5-e["stop"]*.55+e["avgloss"]*2
        shortlist.append((score,p,e))
    shortlist=sorted(shortlist,key=lambda z:z[0],reverse=True)[:300]
    robust=[]
    diagnostics=[]
    for score,p,e in shortlist:
        fs=[ev(f,p) for f in folds]
        ok=all(q and q["n"]>=15 and q["avg"]>0 and q["pf"]>1 for q in fs)
        floor=min((q["avg"] for q in fs if q),default=-999)
        diagnostics.append({"score":score,"params":p,"development":e,"folds":fs,"robust":ok})
        if ok:
            robust_score=score+floor*10-min(np.std([q["avg"] for q in fs]),5)*2
            robust.append((robust_score,p,e,fs))
    diag={"tested":tested,"screened":len(shortlist),"robust_count":len(robust),
          "top_candidates":diagnostics[:20]}
    Path("development_diagnostics.json").write_text(json.dumps(diag,ensure_ascii=False,indent=2),encoding="utf-8")
    if not robust:
        telegram(f"⚠️ TOP-TY-LE-THANG-CAO-GPT vòng 2: chưa có cấu hình robust. Đã test {tested:,}; {len(shortlist)} cấu hình Development có P/L dương. Holdout CHƯA mở.")
        raise RuntimeError("No configuration remained profitable across all Development folds")
    robust.sort(key=lambda z:z[0],reverse=True)
    _,p,de,fs=robust[0]
    # LOCK: Holdout is touched only after the Development winner is fixed.
    ho=ev(hold,p)
    result={"tested":tested,"screened":len(shortlist),"robust_passed":len(robust),
            "locked_params":p,"development":de,"development_folds":fs,"holdout":ho,
            "fee_pct":FEE*100,"target_net_pct":NET_TARGET*100,"data_max":str(x.date.max().date())}
    Path("result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("LOCKED_RESULT",json.dumps(result,ensure_ascii=False))
    h=ho or {}
    telegram("🏆 TOP-TY-LE-THANG-CAO-GPT — KẾT QUẢ KHÓA\n"
      f"DEV: {de['n']} lệnh | ≥2%: {de['win2']:.1f}% | Có lãi: {de['positive']:.1f}% | TB: {de['avg']:+.2f}% | Stop: {de['stop']:.1f}% | PF: {de['pf']:.2f}\n"
      f"HOLDOUT: {h.get('n',0)} lệnh | ≥2%: {h.get('win2',float('nan')):.1f}% | Có lãi: {h.get('positive',float('nan')):.1f}% | TB: {h.get('avg',float('nan')):+.2f}% | Stop: {h.get('stop',float('nan')):.1f}% | PF: {h.get('pf',float('nan')):.2f}\n"
      f"Grid {tested:,} | Robust {len(robust)} | phí 0.4%")
if __name__=="__main__":main()
