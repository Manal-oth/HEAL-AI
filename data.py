"""Real-data loaders for the files supplied with the package (no synthetic data anywhere)."""
from __future__ import annotations
import glob, os, warnings
import numpy as np
import pandas as pd

PHYSIO_VARS = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2",
               "AST","BUN","Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct","Glucose","Lactate",
               "Magnesium","Phosphate","Potassium","Bilirubin_total","TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets"]
PHYSIO_STATIC = ["Age","Gender","Unit1","Unit2","HospAdmTime"]

def physionet_feature_names():
    names = [f"{v}_{s}" for v in PHYSIO_VARS for s in ("last","mean","min","max","obsfrac")]
    return names + PHYSIO_STATIC

def load_physionet2019(raw_dir: str, window_h: int = 12, cache: str | None = None):
    """Patient-level task: from the first `window_h` ICU hours, predict whether SepsisLabel turns 1 later.

    SepsisLabel in the challenge data is already shifted to start 6 h before clinical onset.
    Excluded: stays with <= window_h hours, and patients already labelled septic inside the window.
    Returns X (n,d) with NaN for missing, y (n,), hospital (n,) in {0:setA,1:setB}, ids (n,).
    """
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["hosp"], z["ids"]
    files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.psv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .psv files under {raw_dir}; run scripts/prepare_data.py first")
    X, y, hosp, ids = [], [], [], []
    stats = dict(files=len(files), short_stay=0, septic_in_window=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for f in files:
            d = pd.read_csv(f, sep="|")
            if d["ICULOS"].max() <= window_h:
                stats["short_stay"] += 1; continue
            w = d[d["ICULOS"] <= window_h]
            if len(w) == 0:
                stats["short_stay"] += 1; continue
            if w["SepsisLabel"].max() == 1:
                stats["septic_in_window"] += 1; continue
            lab = int(d.loc[d["ICULOS"] > window_h, "SepsisLabel"].max() == 1)
            v = w[PHYSIO_VARS].to_numpy(dtype=float)
            obs = ~np.isnan(v)
            last = np.array([v[obs[:, j], j][-1] if obs[:, j].any() else np.nan for j in range(v.shape[1])])
            feats = np.stack([last, np.nanmean(v, 0), np.nanmin(v, 0), np.nanmax(v, 0), obs.mean(0)], 1).reshape(-1)
            st = d.iloc[0][PHYSIO_STATIC].to_numpy(dtype=float)
            X.append(np.concatenate([feats, st])); y.append(lab)
            hosp.append(0 if "setA" in f else 1); ids.append(os.path.basename(f)[:-4])
    X, y, hosp, ids = np.array(X), np.array(y), np.array(hosp), np.array(ids)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, X=X, y=y, hosp=hosp, ids=ids, stats=str(stats))
    print("physionet2019 cohort:", stats, "n=", len(y), "pos=", int(y.sum()))
    return X, y, hosp, ids

# --- MIMIC-III clinical database DEMO (100 patients) ---------------------------------------------
_VITALS = {  # itemids (CareVue + MetaVision)
    "HR": [211, 220045], "SBP": [51, 455, 220179, 220050], "DBP": [8368, 8441, 220180, 220051],
    "MAP": [52, 456, 220181, 224322], "RR": [618, 615, 220210, 224690], "TEMPF": [678, 223761],
    "TEMPC": [676, 223762], "SPO2": [646, 220277], "GCS_E": [184, 220739], "GCS_V": [723, 223900],
    "GCS_M": [454, 223901],
}

def load_mimic3_demo(raw_dir: str, window_h: int = 24, cache: str | None = None):
    """In-hospital mortality from the first `window_h` h after first ICU admission (demo subset only)."""
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["ids"], list(z["names"])
    root = raw_dir
    for cand in glob.glob(os.path.join(raw_dir, "**", "ADMISSIONS.csv"), recursive=True):
        root = os.path.dirname(cand); break
    rd = lambda n, **k: pd.read_csv(os.path.join(root, n), **k)
    adm, pat, icu = rd("ADMISSIONS.csv"), rd("PATIENTS.csv"), rd("ICUSTAYS.csv")
    for c in ("admittime",): adm[c] = pd.to_datetime(adm[c])
    pat["dob"] = pd.to_datetime(pat["dob"]); icu["intime"] = pd.to_datetime(icu["intime"])
    first = icu.sort_values("intime").groupby("hadm_id").first().reset_index()[["hadm_id", "icustay_id", "intime"]]
    df = adm.merge(first, on="hadm_id").merge(pat[["subject_id", "gender", "dob"]], on="subject_id")
    df = df.sort_values("admittime").groupby("subject_id").first().reset_index()  # one admission/patient
    df["age"] = ((df["admittime"].dt.year - df["dob"].dt.year)).clip(upper=90)
    ids_item = {i: k for k, v in _VITALS.items() for i in v}
    ce = pd.read_csv(os.path.join(root, "CHARTEVENTS.csv"), usecols=["icustay_id", "itemid", "charttime", "valuenum"])
    ce = ce[ce["itemid"].isin(ids_item) & ce["icustay_id"].isin(df["icustay_id"])].copy()
    ce["charttime"] = pd.to_datetime(ce["charttime"]); ce["var"] = ce["itemid"].map(ids_item)
    ce = ce.merge(df[["icustay_id", "intime"]], on="icustay_id")
    ce = ce[(ce["charttime"] >= ce["intime"]) & (ce["charttime"] <= ce["intime"] + pd.Timedelta(hours=window_h))]
    ce.loc[ce["var"] == "TEMPF", "valuenum"] = (ce["valuenum"] - 32) * 5 / 9
    ce["var"] = ce["var"].replace({"TEMPF": "TEMP", "TEMPC": "TEMP"})
    agg = ce.groupby(["icustay_id", "var"])["valuenum"].agg(["mean", "min", "max"]).unstack("var")
    agg.columns = [f"{v}_{s}" for s, v in agg.columns]
    df = df.merge(agg, left_on="icustay_id", right_index=True, how="left")
    df["male"] = (df["gender"] == "M").astype(float)
    for t in ("EMERGENCY", "ELECTIVE", "URGENT"):
        df[f"adm_{t}"] = (df["admission_type"] == t).astype(float)
    names = ["age", "male"] + [f"adm_{t}" for t in ("EMERGENCY", "ELECTIVE", "URGENT")] + sorted(agg.columns)
    X = df[names].to_numpy(dtype=float); y = df["hospital_expire_flag"].to_numpy(dtype=int)
    ids = df["subject_id"].to_numpy()
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, X=X, y=y, ids=ids, names=np.array(names))
    return X, y, ids, names

def inspect_mimic4_text(csv_path: str) -> dict:
    """The supplied 'MIMIC-IV' file is a discharge-note summarisation table, not admissions/labs/vitals."""
    d = pd.read_csv(csv_path)
    return {"rows": int(len(d)), "columns": list(d.columns),
            "usable_for_paper_temporal_validation": False,
            "reason": "free-text discharge notes + generated summaries only; no admissions, outcomes, labs or vitals"}
