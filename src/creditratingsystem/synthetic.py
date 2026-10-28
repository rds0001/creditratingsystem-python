"""Offline, explicitly driven synthetic credit worlds with separated data layers."""
from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np

from .core import CalcResult, _fail, _fingerprint, _result

_LAYERS = ("observable_inputs", "latent_truth", "frozen_predictions", "decision_history", "observed_outcomes")
_TIERS = {
    "tiny": dict(entities=64, years=3, macro_months=36, paths=512, steps=24, applications=320, households=160, persons=240, application_months=12, account_months=24, pools=2, pool_assets=80, subjects=500),
    "standard": dict(entities=3000, years=10, macro_months=240, paths=16384, steps=60, applications=50000, households=20000, persons=30000, application_months=36, account_months=60, pools=12, pool_assets=30000, subjects=60000),
    "stress": dict(entities=20000, years=15, macro_months=360, paths=131072, steps=120, applications=500000, households=200000, persons=300000, application_months=84, account_months=120, pools=100, pool_assets=100000, subjects=600000),
}
_FACTORS = ("activity", "employment", "rate", "property", "transport", "credit", "fx", "energy")


def _iso(value):
    try:
        d = date.fromisoformat(value)
        if d.isoformat() != value: raise ValueError()
        return d
    except (ValueError, TypeError): _fail("SYNTHETIC_DATE", "Require ISO YYYY-MM-DD dates")


def _month(d, n, end=False):
    year, month = divmod(d.year*12+d.month-1+n, 12); month += 1
    return date(year, month, monthrange(year, month)[1] if end else min(d.day, monthrange(year, month)[1]))


def _json(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _sigmoid(x):
    return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()


def _spec(specification):
    if not isinstance(specification, dict) or not isinstance(specification.get("tier"),str) or specification.get("tier") not in _TIERS:
        _fail("SYNTHETIC_SPEC", "Declare tiny, standard or stress tier")
    s = dict(specification)
    if s.get("dgp_family") not in ("well_specified", "misspecified"):
        _fail("SYNTHETIC_SPEC", "Declare well_specified or misspecified DGP")
    _iso(s.get("start_date")); _iso(s.get("observation_cutoff"))
    if s["observation_cutoff"] < s["start_date"]: _fail("SYNTHETIC_SPEC", "Cutoff precedes world start")
    return s, _TIERS[s["tier"]]


class _Draws:
    """Read supplied uint32-LE uniforms by stable row/component index; no RNG."""
    def __init__(self, directory):
        self.root = Path(directory)
        try: self.manifest = json.loads(_resource_path(self.root,"manifest.json").read_text())
        except (OSError, ValueError): _fail("SYNTHETIC_INNOVATIONS", "Readable innovation manifest required")
        if not isinstance(self.manifest,dict) or self.manifest.get("format") != "creditratingsystem-u32le-v1" or not isinstance(self.manifest.get("streams"),dict):
            _fail("SYNTHETIC_INNOVATIONS", "Unsupported shared innovation format")
        self.arrays = {}

    def row(self, stream, index, minimum):
        if stream not in self.arrays:
            d = self.manifest.get("streams", {}).get(stream)
            if not isinstance(d, dict) or not isinstance(d.get("file"),str) or not isinstance(d.get("sha256"),str) or not isinstance(d.get("shape"), list) or len(d["shape"]) != 2 or any(type(x) is not int or x<=0 for x in d["shape"]):
                _fail("SYNTHETIC_INNOVATIONS", "Missing indexed innovation stream", {"stream": stream})
            path = _resource_path(self.root,d["file"])
            if Path(d["file"]).name != d["file"] or not path.is_file() or path.stat().st_size != math.prod(d["shape"])*4 or _sha(path) != d["sha256"]:
                _fail("SYNTHETIC_INNOVATIONS", "Innovation file size/hash mismatch")
            self.arrays[stream] = np.memmap(path, mode="r", dtype="<u4", shape=tuple(d["shape"]))
        a = self.arrays[stream]
        if index < 0 or index >= len(a) or a.shape[1] < minimum:
            _fail("SYNTHETIC_INNOVATIONS", "Innovation stream capacity too small", {"stream": stream, "index": index})
        return (np.asarray(a[index], dtype=np.float64)+.5)/4294967296.0

    def normals(self, stream, index, count):
        u = self.row(stream, index, 2*((count+1)//2)); z = []
        for j in range(0, 2*((count+1)//2), 2):
            radius = math.sqrt(-2*math.log(float(u[j]))); angle = 2*math.pi*float(u[j+1])
            z.extend((radius*math.cos(angle), radius*math.sin(angle)))
        return z[:count]


class _Writer:
    def __init__(self, output_directory, world_type, specification, innovations, chunk_rows, memory_budget_mb):
        if type(chunk_rows) is not int or not 1 <= chunk_rows <= 50000 or type(memory_budget_mb) is not int or memory_budget_mb < 32:
            _fail("SYNTHETIC_BUDGET", "Require chunk_rows 1..50000 and memory budget >=32MiB")
        estimated_mb = 32+chunk_rows*.004+_TIERS[specification["tier"]]["entities"]*.002
        if memory_budget_mb < estimated_mb:
            _fail("SYNTHETIC_BUDGET", "Declared working-memory budget is below conservative stream/index estimate", {"estimated_working_mb":estimated_mb})
        self.root = Path(output_directory)
        if self.root.exists() and any(self.root.iterdir()): _fail("SYNTHETIC_OUTPUT", "Output directory must be absent or empty")
        self.root.mkdir(parents=True, exist_ok=True)
        for layer in _LAYERS: (self.root/layer).mkdir(exist_ok=True)
        self.tables = {}; self.files = {}; self.started = time.monotonic(); self.chunk_rows = chunk_rows
        self.manifest = {"schema_version":"1.0", "world_type":world_type, "generator_version":"1.0",
                         "data_origin":"SYNTHETIC", "specification":specification,
                         "innovation_manifest_hash":_fingerprint(innovations.manifest) if innovations else None,
                         "chunk_rows":chunk_rows, "memory_budget_mb":memory_budget_mb,
                         "estimated_working_mb":estimated_mb,
                         "layers":list(_LAYERS), "tables":{}, "binary_resources":{},
                         "limitations":["Synthetic engineering evidence does not establish real portfolio performance."]}

    def add(self, layer, table, row):
        key = layer+"/"+table
        if key not in self.files:
            path = self.root/layer/(table+".jsonl.gz")
            raw = path.open("xb"); f = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)
            self.files[key] = (f, raw)
            self.tables[key] = {"file":str(path.relative_to(self.root)), "rows":0, "columns":list(row)}
        if list(row) != self.tables[key]["columns"]:
            _fail("SYNTHETIC_SCHEMA", "Table columns changed during generation", {"table": key})
        try: payload = (_json(row)+"\n").encode("utf-8")
        except (TypeError, ValueError): _fail("SYNTHETIC_NUMERICAL", "Nonfinite or non-JSON generated value")
        self.files[key][0].write(payload); self.tables[key]["rows"] += 1
        if self.tables[key]["rows"] % self.chunk_rows == 0: self.files[key][0].flush()

    def finish(self, additional=None):
        for f, raw in self.files.values(): f.close(); raw.close()
        for key, d in self.tables.items(): d["sha256"] = _sha(self.root/d["file"])
        self.manifest["tables"] = self.tables
        self.manifest["actual_rows"] = sum(t["rows"] for t in self.tables.values())
        self.manifest["runtime_seconds"] = time.monotonic()-self.started
        if additional: self.manifest.update(additional)
        (self.root/"manifest.json").write_text(json.dumps(self.manifest, indent=2)+"\n", encoding="utf-8")
        return _result({"directory":str(self.root), "manifest":self.manifest}, "synthetic.generate_"+self.manifest["world_type"],
                       "synthetic_world", provenance={"data_origin":"SYNTHETIC", "parameter_origin":"SYNTHETIC_REFERENCE"})


def _world(world):
    if isinstance(world, CalcResult): world = world.value
    try: root = Path(world["directory"] if isinstance(world, dict) else world)
    except (TypeError,KeyError): _fail("SYNTHETIC_WORLD", "Supply a generated result or world directory")
    path=_resource_path(root,"manifest.json")
    try: manifest = json.loads(path.read_text())
    except (OSError, ValueError): _fail("SYNTHETIC_WORLD", "Readable generated world required")
    if not isinstance(manifest,dict) or manifest.get("schema_version") != "1.0" or manifest.get("data_origin") != "SYNTHETIC" or not isinstance(manifest.get("tables"),dict) or not isinstance(manifest.get("specification"),dict) or not isinstance(manifest.get("binary_resources"),dict): _fail("SYNTHETIC_WORLD", "Invalid world manifest")
    return root, manifest


def _resource_path(root, relative):
    if not isinstance(relative,str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        _fail("SYNTHETIC_WORLD", "Resource path must be relative and contained in its world")
    path=Path(root)/relative
    if not path.resolve().is_relative_to(Path(root).resolve()) or not path.is_file():
        _fail("SYNTHETIC_WORLD", "Resource is missing or escapes its world directory")
    return path


def _table(world, layer, table, check_hashes=True):
    root, manifest = _world(world); d = manifest["tables"].get(layer+"/"+table)
    if d is None: return
    if not isinstance(d,dict) or d.get("file") != layer+"/"+table+".jsonl.gz": _fail("SYNTHETIC_WORLD", "Invalid table resource path")
    path = _resource_path(root,d["file"])
    if path.resolve().parent.parent != root.resolve(): _fail("SYNTHETIC_WORLD", "Invalid table path")
    if check_hashes and _sha(path)!=d.get("sha256"): _fail("SYNTHETIC_WORLD", "Source table checksum mismatch")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f: yield json.loads(line)


def _linked(world, specification, world_type=None):
    root, manifest = _world(world)
    if world_type and manifest["world_type"] != world_type:
        _fail("SYNTHETIC_LINK", "Unexpected source world type")
    if any(manifest["specification"].get(k)!=specification.get(k) for k in ("tier","start_date","observation_cutoff")):
        _fail("SYNTHETIC_LINK", "Source and output tier/start/cutoff must agree")
    return root,manifest


def generate_macro_reference_world(specification, innovations, *, output_directory, chunk_rows=1000, memory_budget_mb=256):
    """Generate correlated latent factors, lagged releases and a streamed forecast cube."""
    s, t = _spec(specification); draws = _Draws(innovations)
    for stream,rows,width in (("macro_history",t["macro_months"],10),("macro_revision",t["macro_months"],8),("macro_paths",t["paths"]*t["steps"],10)): draws.row(stream,rows-1,width)
    w = _Writer(output_directory,"macro_reference_world",s,draws,chunk_rows,memory_budget_mb)
    start, cutoff = _iso(s["start_date"]), _iso(s["observation_cutoff"])
    rho = [.85,.8,.75,.7,.9,.65,.6,.5]; state = [0.0]*8; indices = [100.0]*8
    for m in range(t["macro_months"]):
        z = draws.normals("macro_history",m,9); u = draws.row("macro_revision",m,8)
        regime = "stress" if s["dgp_family"] == "misspecified" and m >= int(.7*t["macro_months"]) else "baseline"
        for j in range(8):
            shock = .25*z[0]+math.sqrt(.9375)*z[j+1]
            state[j] = rho[j]*state[j]+math.sqrt(1-rho[j]**2)*shock + (-.3 if regime=="stress" and j in (0,3,4) else 0)
            indices[j] *= math.exp(.0015+.002*state[j])
        observed = _month(start,m,True)
        values = [indices[0], min(.3,max(.005,.06-.008*state[1])), min(.3,max(-.01,.03+.004*state[2])), indices[3], indices[4], min(.4,max(.001,.02-.003*state[5])), math.exp(.08*state[6]), indices[7]]
        w.add("latent_truth","macro_factors",dict(period=m,date=observed.isoformat(),regime=regime,**dict(zip(_FACTORS,values))))
        for j, name in enumerate(_FACTORS):
            for revision, lag, noise in (("initial",30,(float(u[j])-.5)*.01),("revised",90,0.0)):
                known = observed+timedelta(days=lag)
                if known <= cutoff:
                    w.add("observable_inputs","macro_releases",dict(release_id=f"macro-{m:04d}-{j}-{revision}",series=name,observed_at=observed.isoformat(),known_at=known.isoformat(),revision=revision,value=values[j]*(1+noise),unit="decimal_rate" if j in (1,2,5) else "index" if j in (0,3,4,7) else "fx_ratio"))
    cube_path = w.root/"latent_truth"/"scenario_cube.f64le"
    # One path at a time; memory is independent of total scenario-cube size.
    with cube_path.open("xb") as f:
        for path in range(t["paths"]):
            x = state.copy()
            for step in range(t["steps"]):
                z = draws.normals("macro_paths",path*t["steps"]+step,9)
                x = [rho[j]*x[j]+math.sqrt(1-rho[j]**2)*(.25*z[0]+math.sqrt(.9375)*z[j+1]) for j in range(8)]
                f.write(struct.pack("<8d",*x))
    w.manifest["binary_resources"]["scenario_cube"] = {"file":str(cube_path.relative_to(w.root)),"dtype":"float64-little-endian","shape":[t["paths"],t["steps"],8],"order":"path,month,factor","factors":list(_FACTORS),"origin":_month(start,t["macro_months"]-1,True).isoformat(),"sha256":_sha(cube_path)}
    return w.finish({"dgp":{"rho":rho,"common_innovation_loading":.25,"factor_stationary_variance_baseline":1,"release_lags_days":[30,90],"forecast_history_separation":"scenario dates strictly after final macro history date"}})


def _splits(group, period, periods):
    entity = "train" if group % 10 < 6 else "calibration" if group % 10 < 8 else "test"
    temporal = "train" if period < .6*periods else "calibration" if period < .8*periods else "test"
    return entity, temporal, entity if entity == temporal else "excluded"


def generate_obligor_panel(macro_world, specification, innovations, *, output_directory, chunk_rows=1000, memory_budget_mb=256):
    """Generate linked statements, independent opinions, latent hazards and defaults.

    Defaults stop subsequent financial statements. The latent Merton asset values
    generate observable equity prices using the option equation, independently of
    the package's structural estimation routines. Forward default labels remain
    separate from lagged, observable financial data.
    """
    s,t = _spec(specification); draws = _Draws(innovations)
    for stream,rows,width in (("obligor_static",t["entities"],16),("obligor_year",t["entities"]*t["years"],16),("obligor_groups",(t["entities"]+3)//4*t["years"],2),("country_transfer",6*t["years"],2)): draws.row(stream,rows-1,width)
    _linked(macro_world,s,"macro_reference_world")
    macro = list(_table(macro_world,"latent_truth","macro_factors"))
    if len(macro) < 12*t["years"]: _fail("SYNTHETIC_LINK", "Macro history must cover all financial years")
    w = _Writer(output_directory,"obligor_panel",s,draws,chunk_rows,memory_budget_mb)
    start = _iso(s["start_date"]); cutoff = _iso(s["observation_cutoff"])
    transfer = {}
    for c in range(6):
        for y in range(t["years"]):
            tu=draws.row("country_transfer",c*t["years"]+y,2)
            probability=.005+.04*max(0,macro[y*12+11]["employment"]-.04)
            transfer[(c,y)]=(probability,float(tu[1])<probability)
            w.add("latent_truth","country_transfer",dict(country_id="country-"+str(c),period=y,transfer_pd=probability,transfer_event=transfer[(c,y)][1],currency_scope="foreign_currency"))
    for i in range(t["entities"]):
        u = draws.row("obligor_static",i,16); oid = f"obligor-{i:06d}"; group = i//4
        country = "country-"+str(i%6); sector = ("industry","trade","property","transport","energy","services")[i%6]
        assets0 = math.exp(13+float(u[0])*7); leverage = .15+.65*float(u[1]); sigma = .12+.3*float(u[2])
        w.add("observable_inputs","obligors",dict(obligor_id=oid,group_id=f"group-{group:06d}",country_id=country,sector=sector,currency="EUR",established_at=start.isoformat(),known_at=start.isoformat()))
        if i%4:
            w.add("observable_inputs","support_edges",dict(edge_id="support-"+oid,supporter_id=f"obligor-{4*group:06d}",beneficiary_id=oid,scope="senior_unsecured",capacity=assets0*.1,effective_from=start.isoformat(),effective_to=_month(start,t["years"]*12).isoformat(),known_at=start.isoformat()))
        for y in range(t["years"]):
            observed = _month(start,y*12+11,True); known = observed+timedelta(days=90)
            v = draws.row("obligor_year",i*t["years"]+y,16)
            g = draws.normals("obligor_groups",group*t["years"]+y,1)[0]
            m = macro[min(y*12+11,len(macro)-1)]; cycle = (m["activity"]-100)/20
            assets = assets0*math.exp(.02*y+.07*(float(v[0])-.5))
            debt = assets*leverage; equity = assets-debt; revenue = assets*(.4+float(u[3]))
            margin = .02+.18*float(u[4])+.015*cycle; ebitda = revenue*margin
            depreciation = assets*.025; ebit = ebitda-depreciation; interest = debt*(.025+m["rate"])
            tax = max(0,ebit-interest)*.25; capex = assets*.03; cfads = ebitda-tax-capex
            debt_service = interest+debt*.1
            features_eta = -4.2+3*(leverage-.4)-2*margin-.35*cycle
            true_eta = features_eta if s["dgp_family"] == "well_specified" else features_eta+.6*g+3*(leverage-.5)**2+.7*(margin<.05)
            hazard = 0 if i%32 == 0 else .0001 if i%32 == 1 else _sigmoid(true_eta)
            d1 = (math.log(assets/debt)+(m["rate"]+.5*sigma*sigma))/sigma; d2 = d1-sigma
            phi = lambda x: .5*(1+math.erf(x/math.sqrt(2)))
            market_equity = assets*phi(d1)-debt*math.exp(-m["rate"])*phi(d2)
            equity_vol = sigma*assets*phi(d1)/market_equity
            entity_split,time_split,split = _splits(group,y,t["years"])
            snapshot_id = f"financial-{i:06d}-{y:02d}"
            if known <= cutoff:
                w.add("observable_inputs","financial_statements",dict(snapshot_id=snapshot_id,obligor_id=oid,observed_at=observed.isoformat(),known_at=known.isoformat(),assets=assets,debt=debt,equity=equity,revenue=revenue,ebitda=ebitda,depreciation=depreciation,ebit=ebit,interest=interest,tax=tax,capex=capex,cfads=cfads,debt_service=debt_service,market_equity=market_equity,equity_volatility=equity_vol,risk_free_rate=m["rate"],entity_split=entity_split,time_split=time_split,split=split))
                opinion = min(7,max(1,round(2+4*leverage-8*margin+(float(v[2])-.5)*2)))
                w.add("observable_inputs","external_opinions",dict(opinion_id="opinion-"+snapshot_id,obligor_id=oid,observed_at=observed.isoformat(),known_at=known.isoformat(),grade=opinion,provider="synthetic_independent_opinion",valid_until=(known+timedelta(days=365)).isoformat()))
                w.add("frozen_predictions","obligor_predictions",dict(snapshot_id=snapshot_id,obligor_id=oid,as_of=known.isoformat(),model_version="fixed-corporate-logit-v1",pd=_sigmoid(features_eta),horizon_months=12))
            defaulted = float(v[3]) < hazard
            event = known+timedelta(days=1+int(float(v[4])*364)) if defaulted else None
            w.add("latent_truth","obligor_period",dict(snapshot_id=snapshot_id,obligor_id=oid,period=y,as_of=known.isoformat(),true_pd=hazard,asset_value=assets,asset_volatility=sigma,origination_leverage=leverage,group_factor=g,dgp_stratum="zero_default" if i%32==0 else "low_default" if i%32==1 else "ordinary",potential_default_at=event.isoformat() if event else None))
            transfer_pd,transfer_event=transfer[(i%6,y)]
            w.add("latent_truth","foreign_currency_risk",dict(snapshot_id=snapshot_id,obligor_id=oid,country_id=country,base_pd=hazard,transfer_pd=transfer_pd,independent_union_pd=1-(1-hazard)*(1-transfer_pd),base_event=defaulted,transfer_event=transfer_event,joint_default_event=bool(defaulted or transfer_event),scope="foreign_currency_sensitivity_separate_from_base_labels"))
            if known <= cutoff:
                matured = known+timedelta(days=365) <= cutoff
                event_known = event+timedelta(days=14) if event else None
                w.add("observed_outcomes","obligor_labels",dict(snapshot_id=snapshot_id,obligor_id=oid,window_start=known.isoformat(),window_end=(known+timedelta(days=365)).isoformat(),evaluation_cutoff=cutoff.isoformat(),default_indicator=1 if event_known and event_known<=cutoff else 0 if matured else None,censored=not matured and not (event_known and event_known<=cutoff)))
            if event and event+timedelta(days=14) <= cutoff:
                w.add("observed_outcomes","default_events",dict(event_id="default-"+oid,obligor_id=oid,event_at=event.isoformat(),known_at=(event+timedelta(days=14)).isoformat(),event_type="DEFAULT"))
            # The legal borrower and operating asset are distinct entities.
            kind = ("commercial_property","aircraft","ship","construction_project","leasing")[i%5]
            units = 10+int(float(u[5])*90); occupied = int(units*(.65+.3*float(v[5])))
            if known <= cutoff:
                w.add("observable_inputs","operating_assets",dict(asset_id="asset-"+oid,obligor_id=oid,asset_type=kind,observed_at=observed.isoformat(),known_at=known.isoformat(),market_value=assets*.7,capacity=units,occupied_units=occupied,utilization=occupied/units,contract_end=_month(known,24+int(float(u[6])*120)).isoformat(),gross_rent_or_revenue=revenue,operating_cost=revenue-ebitda,maintenance_reserve=assets*.01,cfads=cfads,debt_service=debt_service,completion_fraction=min(1,(y+1)/3),lessee_id="lessee-"+oid,lessor_id="lessor-"+oid,spc_id="spc-"+oid))
                for tenant in range(3 if kind=="commercial_property" else 1):
                    w.add("observable_inputs","operating_contracts",dict(contract_id=f"operating-contract-{oid}-{y}-{tenant}",asset_id="asset-"+oid,obligor_id=oid,counterparty_id=f"tenant-{oid}-{tenant}" if kind=="commercial_property" else "lessee-"+oid,contract_type="lease" if kind in ("commercial_property","leasing") else "charter" if kind in ("aircraft","ship") else "offtake",known_at=known.isoformat(),effective_from=known.isoformat(),effective_to=_month(known,24+tenant*12+int(float(u[6])*60)).isoformat(),annual_contract_revenue=revenue/(3 if kind=="commercial_property" else 1),occupied_capacity=occupied/(3 if kind=="commercial_property" else 1),currency="EUR"))
                if i%11==0:
                    w.add("observable_inputs","rating_overrides",dict(override_id="override-"+snapshot_id,obligor_id=oid,known_at=known.isoformat(),effective_from=known.isoformat(),effective_to=_month(known,6).isoformat(),grade_before=opinion,grade_after=min(7,opinion+1),reason_code="documented_liquidity_concern",approved_by="synthetic_second_line"))
            if defaulted: break
    return w.finish({"dependencies":{"macro_manifest_hash":_fingerprint(_world(macro_world)[1])},"dgp":{"first_default_absorbing":True,"financial_reporting_lag_days":90,"outcome_reporting_lag_days":14,"split_unit":"economic_group","split_rule":"group modulo10 and forward time60/20/20; cross-cells excluded","structural_equity":"European call on asset value, horizon1year"}})


def generate_facility_workouts(obligor_world, specification, innovations, *, output_directory, chunk_rows=1000, memory_budget_mb=256):
    """Generate facilities and gross recovery/cost ledgers, including open workouts."""
    s,t = _spec(specification); draws = _Draws(innovations)
    draws.row("facilities",t["entities"]-1,16)
    _linked(obligor_world,s,"obligor_panel")
    w = _Writer(output_directory,"facility_workouts",s,draws,chunk_rows,memory_budget_mb)
    start = _iso(s["start_date"]); cutoff = _iso(s["observation_cutoff"])
    defaults = {r["obligor_id"]:r for r in _table(obligor_world,"observed_outcomes","default_events")}
    origin_leverage = {r["obligor_id"]:r["origination_leverage"] for r in _table(obligor_world,"latent_truth","obligor_period")}
    tenor_months = (t["years"]+2)*12
    obligors = _table(obligor_world,"observable_inputs","obligors")
    for i,o in enumerate(obligors):
        u = draws.row("facilities",i,16); limit0 = math.exp(12+6*float(u[0])); badness = .5*float(u[1])+.5*((origin_leverage[o["obligor_id"]]-.15)/.65)
        for k in range(2):
            fid = f"facility-{i:06d}-{k}"; limit = limit0*(1 if k==0 else .4); revolving = k==1
            balance = limit*(.25+.5*float(u[2])) if revolving else limit
            w.add("observable_inputs","facilities",dict(facility_id=fid,obligor_id=o["obligor_id"],product="revolving" if revolving else "term_loan",currency="EUR",origination_at=start.isoformat(),known_at=start.isoformat(),maturity_at=_month(start,tenor_months).isoformat(),original_limit=limit,original_balance=balance,interest_rate=.035+.025*badness,seniority="senior_secured" if k==0 else "senior_unsecured",collateral_id="collateral-"+fid if k==0 else None))
            w.add("observable_inputs","collateral",dict(collateral_id="collateral-"+fid,facility_id=fid,valuation_at=start.isoformat(),known_at=start.isoformat(),value=limit*(1.1+.5*float(u[3])) if k==0 else 0,haircut=.15+.15*badness,lien_rank=1,currency="EUR"))
            event = defaults.get(o["obligor_id"])
            for month in range(t["years"]*12):
                d = _month(start,month,True)
                if d>cutoff or (event and d>=_iso(event["event_at"])): break
                amount = limit*(.25+.5*float(u[2])+.1*math.sin(month/6)) if revolving else balance*max(0,1-month/tenor_months)
                w.add("observable_inputs","facility_exposures",dict(facility_id=fid,observed_at=d.isoformat(),known_at=d.isoformat(),drawn_balance=amount,credit_limit=limit,undrawn=limit-amount))
            if event:
                d = _iso(event["event_at"]); elapsed = (d.year-start.year)*12+d.month-start.month
                prior = limit*(.25+.5*float(u[2])+.1*math.sin(elapsed/6)) if revolving else balance*max(0,1-elapsed/tenor_months)
                ead = prior+max(0,limit-prior)*(.3+.6*badness) if revolving else prior
                recovery_fraction = (.65+.45*float(u[4])-.15*badness) if k==0 else (.05+.6*float(u[4])-.05*badness)
                cost_fraction = .03+.15*badness; rate = .04
                total_pv = 0.; last = None
                for n,delay in enumerate((90,360,900)):
                    rd = d+timedelta(days=delay); last = rd
                    recovery = ead*recovery_fraction*(.2,.3,.5)[n]; cost = ead*cost_fraction/3
                    total_pv += (recovery-cost)/(1+rate)**(delay/365)
                    row = dict(cashflow_id=f"recovery-{fid}-{n}",facility_id=fid,default_event_id=event["event_id"],event_at=rd.isoformat(),known_at=(rd+timedelta(days=7)).isoformat(),recovery_amount=recovery,cost_amount=cost,currency="EUR")
                    w.add("latent_truth","future_recoveries",row)
                    if rd+timedelta(days=7)<=cutoff: w.add("observed_outcomes","recovery_cashflows",row)
                w.add("observed_outcomes","workouts",dict(workout_id="workout-"+fid,facility_id=fid,default_event_id=event["event_id"],default_at=d.isoformat(),known_at=event["known_at"],ead=ead,discount_rate=rate,status="closed" if last+timedelta(days=7)<=cutoff else "open",closed_at=last.isoformat() if last+timedelta(days=7)<=cutoff else None))
                w.add("latent_truth","workout_parameters",dict(facility_id=fid,ead=ead,recovery_fraction=recovery_fraction,cost_fraction=cost_fraction,ultimate_discounted_lgd=1-total_pv/ead,driver=badness))
    # Deterministic economic boundary ledgers are separate from stochastic cohorts.
    for name,ead,recovery,cost in (("negative_lgd",100,120,0),("lgd_above_one",100,0,20),("total_loss",100,0,0),("no_loss",100,100,0)):
        w.add("latent_truth","analytical_workout_fixtures",dict(case_id=name,ead=ead,recovery_at_default=recovery,cost_at_default=cost,expected_raw_lgd=1-(recovery-cost)/ead,constructed=True))
    w.add("latent_truth","constructed_open_workouts",dict(case_id="open_workout",default_at=max(start,cutoff-timedelta(days=30)).isoformat(),evaluation_cutoff=cutoff.isoformat(),ead=100.,observed_recovery=20.,observed_cost=5.,observed_loss_fraction=.85,ultimate_lgd=None,status="open",constructed=True,excluded_from_stochastic_counts=True))
    return w.finish({"dependencies":{"obligor_manifest_hash":_fingerprint(_world(obligor_world)[1])},"dgp":{"workout_lags_days":[90,360,900],"discount_day_basis":365,"raw_lgd_is_not_clipped":True,"observed_recoveries_respect_cutoff":True}})


def generate_retail_reference_world(macro_world, specification, innovations, *, output_directory, chunk_rows=1000, memory_budget_mb=256):
    """Generate households through collections without labelling rejected loans.

    ``selection_mechanism`` is MAR, MNAR or missing_overlap. This changes the
    acceptance mechanism, never the actual shared draws or the hidden outcomes.
    Income is explicitly net; no tax conversion or model fitting is performed.
    """
    s,t = _spec(specification)
    if s.get("selection_mechanism") not in ("MAR","MNAR","missing_overlap"):
        _fail("SYNTHETIC_SPEC", "Declare selection_mechanism MAR, MNAR or missing_overlap")
    draws = _Draws(innovations)
    for stream,rows,width in (("retail_households",t["households"],16),("retail_apps",t["applications"],24),("retail_account",t["applications"]*t["account_months"],8)): draws.row(stream,rows-1,width)
    _linked(macro_world,s,"macro_reference_world")
    w = _Writer(output_directory,"retail_reference_world",s,draws,chunk_rows,memory_budget_mb)
    start = _iso(s["start_date"]); cutoff = _iso(s["observation_cutoff"])
    macro = list(_table(macro_world,"latent_truth","macro_factors"))
    if not macro: _fail("SYNTHETIC_LINK", "Macro factors required")
    products = ("mortgage","consumer_installment","auto_loan","credit_card","overdraft","bnpl","small_business","retail_leasing")
    for h in range(t["households"]):
        u = draws.row("retail_households",h,16); hid = f"household-{h:06d}"
        w.add("observable_inputs","households",dict(household_id=hid,created_at=start.isoformat(),known_at=start.isoformat(),dependants=int(float(u[0])*4),currency="EUR"))
    for p in range(t["persons"]):
        h = p%t["households"]; u = draws.row("retail_households",h,16)
        w.add("observable_inputs","persons",dict(person_id=f"person-{p:06d}",household_id=f"household-{h:06d}",birth_year=start.year-21-int(float(u[1])*45),employment="self_employed" if p%5==0 else "salaried",known_at=start.isoformat()))
        w.add("observable_inputs","household_memberships",dict(membership_id=f"membership-{p:06d}",person_id=f"person-{p:06d}",household_id=f"household-{h:06d}",valid_from=start.isoformat(),valid_to=None,known_at=start.isoformat()))
    funded_count = 0; capacity = {"tiny":160,"standard":24000,"stress":240000}[s["tier"]]
    for i in range(t["applications"]):
        person = i%t["persons"]; h = person%t["households"]; hu = draws.row("retail_households",h,16)
        u = draws.row("retail_apps",i,24); month = i%t["application_months"]
        d = _month(start,month)+timedelta(days=int(float(u[0])*27)); app = f"application-{i:07d}"
        if d>cutoff: continue
        income = math.exp(7.4+1.2*float(hu[2])); expenses = income*(.3+.25*float(hu[3])); debt = income*(.05+.25*float(hu[4]))
        latent = 2*(float(hu[5])-.5); m = macro[min(month,len(macro)-1)]
        product = products[i%len(products)]; term = 240 if product=="mortgage" else 12 if product in ("bnpl","overdraft","credit_card") else 60
        principal = income*(45 if product=="mortgage" else 2 if product=="bnpl" else 8 if product in ("overdraft","credit_card") else 15)
        rate = .045 if product=="mortgage" else .11; periodic = rate/12
        balloon = principal*.15 if product=="auto_loan" else principal*.3 if product=="retail_leasing" else 0.
        installment = (principal-balloon/(1+periodic)**term)*periodic/(-math.expm1(-term*math.log1p(periodic)))
        dti = (installment+debt)/income; bureau_risk = min(1,max(0,.25+.25*latent+.2*(float(u[1])-.5)))
        thin = i%13==0; nohit = i%29==0; stale = i%17==0
        missing_mechanism = ("MCAR","MAR","MNAR")[i%3]
        missing_p = .06 if missing_mechanism=="MCAR" else .03+.15*(income<2500) if missing_mechanism=="MAR" else _sigmoid(-3+latent)
        missing_income = float(u[2])<missing_p
        scored_dti = (installment+debt)/(2500 if missing_income else income)
        score_eta = -4+2.5*scored_dti+2*(.5 if nohit else bureau_risk)+.4*missing_income
        prediction = _sigmoid(score_eta)
        true_eta = score_eta if s["dgp_family"]=="well_specified" else score_eta+.7*latent+2*(dti-.4)**2+3*(m["employment"]-.06)
        true_pd = _sigmoid(true_eta); potential_default = float(u[3])<true_pd
        entity_split,time_split,split = _splits(h,month,t["application_months"])
        w.add("observable_inputs","applications",dict(application_id=app,person_id=f"person-{person:06d}",household_id=f"household-{h:06d}",product=product,application_at=d.isoformat(),known_at=d.isoformat(),requested_amount=principal,term_months=term,currency="EUR",net_monthly_income=None if missing_income else income,monthly_expenses=expenses,existing_debt_service=debt,proposed_installment=installment,income_basis="net",income_missing=missing_income,entity_split=entity_split,time_split=time_split,split=split))
        w.add("observable_inputs","application_parties",dict(application_id=app,person_id=f"person-{person:06d}",role="primary_borrower",household_id=f"household-{h:06d}",known_at=d.isoformat(),joint_and_several=True))
        partner = h+t["households"] if person<t["households"] else h
        if i%2==0 and partner<t["persons"] and partner!=person:
            w.add("observable_inputs","application_parties",dict(application_id=app,person_id=f"person-{partner:06d}",role="co_borrower",household_id=f"household-{h:06d}",known_at=d.isoformat(),joint_and_several=True))
        pull = "bureau-"+app; observed = d-timedelta(days=400 if stale else 7)
        w.add("observable_inputs","bureau_pulls",dict(pull_id=pull,application_id=app,observed_at=observed.isoformat(),known_at=d.isoformat(),no_hit=nohit,thin_file=thin,stale=stale,risk_score=None if nohit else bureau_risk))
        if i%19==0 and d+timedelta(days=45)<=cutoff:
            w.add("observable_inputs","bureau_revisions",dict(revision_id="revision-"+pull,pull_id=pull,observed_at=observed.isoformat(),known_at=(d+timedelta(days=45)).isoformat(),revised_risk_score=min(1,bureau_risk+.1),supersedes="initial",eligible_at_application=False))
        if not nohit:
            for b in range(1 if thin else 3):
                w.add("observable_inputs","bureau_tradelines",dict(tradeline_id=pull+"-"+str(b),pull_id=pull,observed_at=observed.isoformat(),known_at=d.isoformat(),balance=debt*12/(1 if thin else 3),monthly_payment=debt/(1 if thin else 3),status="current" if bureau_risk<.5 else "delinquent"))
        # Cash observations predate application; transfers and reversals have explicit links.
        for j in range(3):
            txd = d-timedelta(days=30*(j+1)); prefix=f"transaction-{app}-{j}"
            for k,kind,amount,link in ((0,"salary",income,None),(1,"living_expense",-expenses,None),(2,"internal_transfer",-100,prefix+"-3"),(3,"internal_transfer",100,prefix+"-2"),(4,"card_charge",-25,prefix+"-5"),(5,"reversal",25,prefix+"-4")):
                w.add("observable_inputs","transactions",dict(transaction_id=prefix+"-"+str(k),application_id=app,household_id=f"household-{h:06d}",account_id=f"external-account-{h:06d}",event_at=txd.isoformat(),known_at=d.isoformat(),category=kind,amount=amount,currency="EUR",linked_transaction_id=link))
        policy_version = "retail-policy-v1" if month<t["application_months"]//2 else "retail-policy-v2"
        acceptance = _sigmoid(2.2-5*prediction-.5*scored_dti-(.7*latent if s["selection_mechanism"]=="MNAR" else 0))
        if s["selection_mechanism"]=="missing_overlap" and scored_dti>.55: acceptance=0.
        if policy_version.endswith("v2"): acceptance*=.9
        accepted = float(u[4])<acceptance; taken = accepted and float(u[5])<.65
        funded = taken and funded_count<capacity; account = "account-"+app if funded else None
        w.add("frozen_predictions","application_predictions",dict(application_id=app,as_of=d.isoformat(),model_version="fixed-retail-logit-v1",pd=prediction,horizon_months=12,feature_source="observable application and bureau only"))
        for stage,outcome in (("acceptance",accepted),("take_up",taken),("funding",funded)):
            w.add("decision_history","decisions",dict(decision_id=app+"-"+stage,application_id=app,stage=stage,decided_at=d.isoformat(),policy_version=policy_version,result=outcome,account_id=account,capacity_exhausted=bool(taken and not funded)))
        default_mob = 1+int(float(u[6])*11) if potential_default else None
        w.add("latent_truth","application_parameters",dict(application_id=app,true_pd=true_pd,latent_frailty=latent,acceptance_probability=acceptance,potential_default_mob=default_mob,potential_loan_outcome=1 if potential_default else 0,observed_income_missing=missing_income,missing_mechanism=missing_mechanism))
        # Rejected/unfunded applicants have no observed own-loan label.
        matured = _month(d,12)<=cutoff
        if not funded:
            w.add("observed_outcomes","application_labels",dict(application_id=app,account_id=None,funded=False,window_end=_month(d,12).isoformat(),evaluation_cutoff=cutoff.isoformat(),own_loan_default=None,censored=False))
            continue
        funded_count+=1
        secured = product in ("mortgage","auto_loan","retail_leasing"); revolving = product in ("credit_card","overdraft")
        w.add("observable_inputs","accounts",dict(account_id=account,application_id=app,customer_id=f"person-{person:06d}",household_id=f"household-{h:06d}",product_id=product,opened_at=d.isoformat(),known_at=d.isoformat(),currency="EUR",original_amount=principal,credit_limit=principal if revolving else 0,contract_months=term,annual_rate=rate,rate_type="floating" if product=="mortgage" else "fixed",reset_month=12 if product=="mortgage" else None,balloon_amount=balloon,collateral_value=principal/.75 if secured else None,first_lien_amount=principal if secured else None))
        balance = principal*(.4 if revolving else 1); cumulative_loss = 0.; default_at = None; close_at = None; first_default_at = None; cured = False; past_due = 0.; first_unpaid_at = None; episode_number=0; episode_default_mob=default_mob
        for mob in range(t["account_months"]):
            date_m = _month(d,mob,True)
            if date_m>cutoff: break
            av = draws.row("retail_account",i*t["account_months"]+mob,8)
            if close_at: break
            event_type = None
            if default_mob == mob:
                default_at=date_m; first_default_at=date_m; event_type="DEFAULT"; cumulative_loss=balance; episode_number=1; episode_default_mob=mob
            elif cured and mob==default_mob+10 and float(u[8])<.5:
                default_at=date_m; event_type="DEFAULT"; cumulative_loss=balance; episode_number+=1; episode_default_mob=mob; cured=False
            elif default_at and mob==episode_default_mob+3: event_type="CHARGEOFF"
            elif default_at and mob==episode_default_mob+6 and float(u[7])<.2: event_type="CURED"; default_at=None; cured=True
            elif not default_at and mob>1 and float(av[0])<.008: event_type="PREPAID"; close_at=date_m
            elif mob==term and not default_at: event_type="CLOSED"; close_at=date_m
            prior=balance; draw=0.; payment=0.; recovery=0.
            if product=="mortgage" and mob==12:
                rate=.02+macro[min(month+mob,len(macro)-1)]["rate"]; periodic=rate/12
                remaining=max(1,term-mob)
                installment=(prior-balloon/(1+periodic)**remaining)*periodic/(-math.expm1(-remaining*math.log1p(periodic)))
            interest=prior*periodic if not default_at else 0.
            if not default_at:
                scheduled=min(prior+interest,installment+past_due) if not revolving else prior+interest if mob%6==0 else prior*.03+interest+past_due
                scheduled=min(prior+interest,scheduled)
                missed=float(av[2])<.04 and not close_at
                payment=0. if missed else scheduled
                if missed:
                    past_due=scheduled
                    if first_unpaid_at is None: first_unpaid_at=date_m.replace(day=1)
                else: past_due=0.; first_unpaid_at=None
                balance=max(0,prior+interest-payment)
                if revolving and mob%6==1 and not close_at:
                    draw=max(0,principal*(.2+.6*float(av[1]))-balance); balance+=draw
            else:
                scheduled=installment
                elapsed=mob-episode_default_mob
                recovery=min(balance,principal*.01*(1-float(hu[5]))) if elapsed>0 else 0.
                balance=max(0,balance-recovery)
            if close_at: payment=prior+interest; balance=0.; past_due=0.; first_unpaid_at=None
            if mob==0:
                balance=prior; interest=0.; payment=0.; draw=0.; recovery=0.; past_due=0.; first_unpaid_at=None; scheduled=installment
            dpd=90+30*(mob-episode_default_mob) if default_at else (date_m-first_unpaid_at).days if first_unpaid_at else 0
            state="CLOSED" if close_at else "DEFAULT" if default_at else "CURED" if cured else "ACTIVE"
            w.add("observable_inputs","account_months",dict(account_id=account,observed_at=date_m.isoformat(),known_at=date_m.isoformat(),months_on_book=mob,opening_balance=prior,balance=balance,credit_limit=principal if revolving else 0,principal_draw=draw,payment=payment,interest=interest,recovery_amount=recovery,annual_rate=rate,scheduled_payment=scheduled,days_past_due=dpd,past_due_amount=min(balance,installment*(dpd//30)) if default_at else past_due,state=state,active_zero_balance=bool(state in ("ACTIVE","CURED") and balance==0),currency="EUR"))
            if event_type and date_m+timedelta(days=14)<=cutoff:
                w.add("observed_outcomes","account_events",dict(event_id=f"event-{account}-{mob}",account_id=account,event_at=date_m.isoformat(),known_at=(date_m+timedelta(days=14)).isoformat(),event_type=event_type,amount=cumulative_loss if event_type=="CHARGEOFF" else 0))
            if default_at and mob>=default_mob:
                episode="collection-"+account+"-"+str(episode_number); elapsed=mob-episode_default_mob
                action="contact" if elapsed<3 else "restructure" if elapsed<6 else "legal_recovery"
                cost=5 if action=="contact" else 20 if action=="restructure" else 40
                w.add("observed_outcomes","collections",dict(episode_id=episode,account_id=account,event_at=date_m.isoformat(),known_at=date_m.isoformat(),months_since_default=elapsed,action=action,recovery_amount=recovery,cost_amount=cost,currency="EUR",open=balance>0))
        observed_default = first_default_at is not None and first_default_at+timedelta(days=14)<=cutoff
        complete = matured or close_at is not None
        w.add("observed_outcomes","application_labels",dict(application_id=app,account_id=account,funded=True,window_end=_month(d,12).isoformat(),evaluation_cutoff=cutoff.isoformat(),own_loan_default=1 if observed_default else 0 if complete else None,censored=not complete and not observed_default))
    return w.finish({"dependencies":{"macro_manifest_hash":_fingerprint(_world(macro_world)[1])},"actual_funded_accounts":funded_count,"dgp":{"selection_mechanism":s["selection_mechanism"],"missingness":["MCAR","MAR","MNAR"],"funded_account_capacity":capacity,"capacity_rule":"first funded candidates in application-ID order","potential_outcomes_are_latent":True,"split_unit":"household","no_implicit_fitting":True}})


def generate_asset_pool_reference_world(worlds, specification, innovations, *, output_directory, chunk_rows=1000, memory_budget_mb=256):
    """Partition generated source assets; observed servicing copies source ledgers.

    Future contractual schedules are a separate, explicitly no-default projection.
    They are never substituted for actual source payments or recoveries.
    """
    s,t = _spec(specification); draws = _Draws(innovations)
    if not isinstance(worlds,(list,tuple)) or not worlds: _fail("SYNTHETIC_LINK", "Supply generated source worlds")
    for source in worlds: _linked(source,s)
    w = _Writer(output_directory,"asset_pool_reference_world",s,draws,chunk_rows,memory_budget_mb)
    start = _iso(s["start_date"]); cutoff = _iso(s["observation_cutoff"])
    totals=[0.]*t["pools"]; pool_known=[start]*t["pools"]; count=0; seen=set(); per_source=max(1,t["pool_assets"]//len(worlds))
    for source in worlds:
        root,manifest=_world(source); selected={}; nsource=0
        for table,idcol,amountcol,datecol in (("accounts","account_id","original_amount","opened_at"),("facilities","facility_id","original_balance","origination_at")):
            for asset in _table(source,"observable_inputs",table):
                if nsource>=per_source or count>=t["pool_assets"]: break
                aid=asset[idcol]
                if aid in seen: _fail("SYNTHETIC_LINK", "Duplicate source asset across worlds")
                seen.add(aid); p=count%t["pools"]; pid=f"pool-{p:04d}"; amount=asset[amountcol]; valid=max(start,_iso(asset[datecol])); expiry=_month(valid,t["account_months"])
                selected[aid]=(pid,valid,expiry); totals[p]+=amount; pool_known[p]=max(pool_known[p],valid)
                w.add("observable_inputs","pool_memberships",dict(membership_id=f"pool-member-{count:07d}",pool_id=pid,source_world_type=manifest["world_type"],source_asset_id=aid,asset_type=table,partition_share=1.,original_principal=amount,valid_from=valid.isoformat(),valid_to=expiry.isoformat(),known_at=valid.isoformat()))
                term=asset.get("contract_months",(t["years"]+2)*12); rate=asset.get("annual_rate",asset.get("interest_rate",.05))/12
                balloon=asset.get("balloon_amount",0.)
                scheduled=(amount-balloon/(1+rate)**term)*rate/(-math.expm1(-term*math.log1p(rate))) if rate else (amount-balloon)/term; opening=amount
                for mob in range(t["account_months"]):
                    d=_month(valid,mob,True); interest=opening*rate if mob else 0.; principal=min(opening,max(0,scheduled-interest)) if mob else 0.
                    if mob==term: principal=opening
                    closing=opening-principal
                    w.add("latent_truth","contractual_asset_cashflows",dict(cashflow_id=f"pool-cash-{count:07d}-{mob:03d}",pool_id=pid,source_asset_id=aid,event_at=d.isoformat(),opening_principal=opening,scheduled_principal=principal,interest=interest,closing_principal=closing,assumption="no_default_no_prepayment_projection"))
                    opening=closing
                count+=1;nsource+=1
        # Stream each real source ledger once; temporal memberships select deliveries.
        for layer,table,idcol,datecol,components in (
            ("observable_inputs","account_months","account_id","observed_at",(("payment","payment"),)),
            ("observed_outcomes","account_events","account_id","event_at",(("event","amount"),)),
            ("observed_outcomes","collections","account_id","event_at",(("recovery","recovery_amount"),("cost","cost_amount"))),
            ("observed_outcomes","recovery_cashflows","facility_id","event_at",(("recovery","recovery_amount"),("cost","cost_amount"))),
            ("observed_outcomes","workouts","facility_id","default_at",(("default_exposure","ead"),))):
            for index,row in enumerate(_table(source,layer,table)):
                aid=row[idcol]
                if aid not in selected: continue
                pid,valid,expiry=selected[aid]; d=_iso(row[datecol]); known=_iso(row["known_at"])
                if not valid<=d<expiry or known>cutoff: continue
                for kind,column in components:
                    w.add("observed_outcomes","servicing_cashflows",dict(cashflow_id=f"source-{aid}-{table}-{index}-{kind}",pool_id=pid,source_asset_id=aid,source_table=table,event_at=d.isoformat(),known_at=known.isoformat(),component="event_"+row["event_type"] if kind=="event" else kind,amount=row[column],currency="EUR"))
    if not count: _fail("SYNTHETIC_LINK", "No generated funded accounts or facilities available")
    for p,total in enumerate(totals):
        pid=f"pool-{p:04d}"
        w.add("observable_inputs","pools",dict(pool_id=pid,currency="EUR",created_at=start.isoformat(),known_at=pool_known[p].isoformat(),initial_principal=total,notional_basis="sum_original_assigned_principal",revolving=True,clean_up_call_fraction=.1))
        for rank,(name,share) in enumerate((("senior",.7),("mezzanine",.15),("junior",.1),("equity",.05)),1):
            w.add("observable_inputs","tranches",dict(tranche_id=pid+"-"+name,pool_id=pid,seniority_rank=rank,original_principal=total*share,coupon_rate=.03+.015*(rank-1),known_at=pool_known[p].isoformat()))
    return w.finish({"actual_source_assets":count,"asset_capacity":t["pool_assets"],"dependencies":{"source_manifest_hashes":[_fingerprint(_world(x)[1]) for x in worlds]},"dgp":{"membership_partition_share":1,"source_reuse":False,"tranches_per_pool":4,"servicing_origin":"actual source ledgers","contractual_projection":"no default or prepayment; latent only"}})


def generate_system_event_reference_world(worlds, specification, innovations, *, output_directory, chunk_rows=1000, memory_budget_mb=256):
    """Generate dated deliveries plus independently specified state checkpoints."""
    s,t=_spec(specification); draws=_Draws(innovations)
    if not isinstance(worlds,(list,tuple)) or not worlds: _fail("SYNTHETIC_LINK", "Supply generated subject worlds")
    for source in worlds: _linked(source,s)
    w=_Writer(output_directory,"system_event_reference_world",s,draws,chunk_rows,memory_budget_mb)
    start=_iso(s["start_date"]); cutoff=_iso(s["observation_cutoff"]); count=0
    for version,offset,intercept in (("v1",0,100.),("v2",120,105.)):
        w.add("observable_inputs","model_versions",dict(model_version=version,model_role="constructed_score_demonstration",algorithm="fixed_score_schedule",parameter_origin="SYNTHETIC_REFERENCE",intercept=intercept,known_at=start.isoformat(),effective_from=(start+timedelta(days=offset)).isoformat()))
        w.add("observable_inputs","policy_versions",dict(policy_version="policy-"+version,known_at=start.isoformat(),effective_from=(start+timedelta(days=offset)).isoformat(),minimum_score=80. if version=="v1" else 90.,policy_scope="constructed_checkpoint_fixture"))
    w.add("observable_inputs","calibration_versions",dict(calibration_version="calibration-v1",known_at=start.isoformat(),effective_from=start.isoformat(),transform="identity_score",intercept=0.,slope=1.,estimated=False))
    source_models=set(); seen_subjects=set()
    for source in worlds:
        source_root,manifest=_world(source)
        source_model={"obligor_panel":"fixed-corporate-logit-v1","retail_reference_world":"fixed-retail-logit-v1"}.get(manifest["world_type"])
        if source_model and source_model not in source_models:
            w.add("observable_inputs","source_model_versions",dict(model_version=source_model,source_world_type=manifest["world_type"],model_role="frozen_pd_demonstration",known_at=start.isoformat(),estimated=False))
            source_models.add(source_model)
        for table,idcol in (("obligors","obligor_id"),("accounts","account_id"),("facilities","facility_id"),("pools","pool_id")):
            for subject in _table(source,"observable_inputs",table):
                if count>=t["subjects"]: break
                sid=subject[idcol]; source_known=_iso(subject.get("known_at",start.isoformat())); origin=max(start,source_known)
                if sid in seen_subjects: _fail("SYNTHETIC_LINK", "Subject occurs in more than one supplied source")
                seen_subjects.add(sid)
                w.add("observable_inputs","subjects",dict(subject_id=sid,subject_type=table,source_world_type=manifest["world_type"],source_snapshot_hash=_fingerprint(subject),known_at=origin.isoformat()))
                w.add("observable_inputs","source_"+table,dict(subject,snapshot_hash=_fingerprint(subject)))
                # Effective-time chronology and delivery-time chronology deliberately differ.
                schedule=((0,0,"SUBJECT_CREATED","v1",100.,True),(30,30,"SCORE_UPDATED","v1",110.,True),(30,60,"LATE_CORRECTION","v1",105.,True),(45,45,"FAILED_BATCH","v1",999.,False),(90,90,"GUARANTEE_EXPIRED","v1",0.,True),(120,120,"MODEL_SWITCH","v2",105.,True))
                for j,(effective,delay,kind,version,value,valid) in enumerate(schedule):
                    occurred=origin+timedelta(days=effective); known=origin+timedelta(days=delay)
                    if known>cutoff: continue
                    eid=f"system-{count:06d}-{j}"
                    row=dict(delivery_id=eid+"-delivery-1",event_id=eid,subject_id=sid,event_type=kind,effective_at=occurred.isoformat(),known_at=known.isoformat(),model_version=version,policy_version="policy-"+version,calibration_version="calibration-v1",value=value,valid=valid,source_snapshot_hash=_fingerprint(subject),supersedes_event_id=f"system-{count:06d}-1" if kind=="LATE_CORRECTION" else None)
                    w.add("observed_outcomes","system_events",row)
                    if j==1:
                        row=dict(row,delivery_id=eid+"-delivery-2")
                        w.add("observed_outcomes","system_events",row)
                for asof,knowledge,score,support,version in ((35,35,110.,100.,"v1"),(35,65,105.,100.,"v1"),(95,95,105.,0.,"v1"),(125,125,105.,0.,"v2")):
                    if origin+timedelta(days=knowledge)>cutoff: continue
                    w.add("latent_truth","independent_checkpoints",dict(checkpoint_id=f"checkpoint-{count:06d}-{asof}-{knowledge}",subject_id=sid,as_of=(origin+timedelta(days=asof)).isoformat(),known_at=(origin+timedelta(days=knowledge)).isoformat(),expected_score=score,expected_support_capacity=support,expected_model_version=version,expected_policy_version="policy-"+version,expected_calibration_version="calibration-v1",expected_duplicate_effect=0,expected_failed_batch_effect=0,oracle="closed-form authored state schedule; no replay call"))
                count+=1
    if not count: _fail("SYNTHETIC_LINK", "No existing subjects in source worlds")
    return w.finish({"actual_subjects":count,"dependencies":{"source_manifest_hashes":[_fingerprint(_world(x)[1]) for x in worlds]},"dgp":{"duplicate_semantics":"event_id idempotent; delivery_id distinct","failure_semantics":"invalid batch has no state effect","checkpoint_oracle":"authored independently of event replay"}})


def generate_challenge_cases(case_ids=None, context=None):
    """Return constructed analytical cases; these are not random cohort statistics."""
    cases = [
        dict(case_id="complete_separation",inputs={"x":[-2,-1,1,2],"y":[0,0,1,1]},expected={"unpenalized_finite_mle":False}),
        dict(case_id="zero_default",inputs={"defaults":0,"observations":100,"alpha":.05},expected={"exact_one_sided_upper_pd":1-.05**.01}),
        dict(case_id="spiegelhalter_half",inputs={"pd":[.5]*4,"outcome":[0,1,0,1]},expected={"variance":0,"normal_z_defined":False}),
        dict(case_id="singular_covariance",inputs={"covariance":[[1,1],[1,1]]},expected={"rank":1,"determinant":0}),
        dict(case_id="cyclic_support",inputs={"edges":[["a","b"],["b","a"]]},expected={"acyclic":False}),
        dict(case_id="transfer_extremes",inputs={"base_pd":.1,"transfer_pd":[0,1]},expected={"independent_union_pd":[.1,1.]}),
        dict(case_id="censored_target",inputs={"origin":"2020-01-01","horizon_months":12,"cutoff":"2020-06-30","events":[]},expected={"binary_target":None,"censored":True}),
        dict(case_id="raw_lgd_bounds",inputs={"ead":[100,100],"recovery":[120,0],"cost":[0,20]},expected={"raw_lgd":[-.2,1.2]}),
        dict(case_id="cohort_100",inputs={"initial_active":100,"defaulted":10,"closed":20,"active_zero_balance":1},expected={"surviving_active":70,"cumulative_pd":.1,"denominator":100}),
        dict(case_id="future_feature",inputs={"as_of":"2020-01-31","feature_known_at":"2020-02-01"},expected={"feature_eligible":False}),
    ]
    if case_ids is not None:
        if not isinstance(case_ids,(list,tuple)) or len(set(case_ids))!=len(case_ids) or any(x not in {r["case_id"] for r in cases} for x in case_ids): _fail("SYNTHETIC_CASE", "Unknown or duplicate challenge case")
        cases=[r for r in cases if r["case_id"] in case_ids]
    return _result({"cases":cases,"constructed":True,"oracle_origin":"independent analytical identities","data_origin":"SYNTHETIC"},"synthetic.generate_challenge_cases","synthetic_challenges",context,provenance={"data_origin":"SYNTHETIC","parameter_origin":"SYNTHETIC_REFERENCE"})


def validate_synthetic_world(world, check_hashes=True, context=None):
    """Audit serialized rows, temporal knowledge, economic identities and layers."""
    if type(check_hashes) is not bool: _fail("SYNTHETIC_INPUT", "check_hashes must be boolean")
    root,manifest=_world(world); issues=[]; rows=0; cutoff=manifest["specification"]["observation_cutoff"]
    counts={}; actual_defaults=0; funded=set(); obligations=set(); facilities=set(); account_ids=set(); groups={}; pool_sums={}; tranche_sums={}; sources=set()
    design_pd_sum=0.; latent_default_count=0; risk_observations=0; actual_retail_defaults=set()
    def issue(code,table,index):
        if len(issues)<100: issues.append(dict(code=code,table=table,row=index))
    forbidden={"true_pd","latent_frailty","potential_default_mob","potential_loan_outcome","ultimate_discounted_lgd","potential_default_at","asset_volatility","origination_leverage","missing_mechanism"}
    registries = {}
    definitions = {"obligors":"obligor_id","households":"household_id","persons":"person_id","applications":"application_id","bureau_pulls":"pull_id","accounts":"account_id","facilities":"facility_id","pools":"pool_id","subjects":"subject_id","model_versions":"model_version","policy_versions":"policy_version","calibration_versions":"calibration_version"}
    for table,column in definitions.items():
        if "observable_inputs/"+table in manifest["tables"]:
            ids=set()
            for index,row in enumerate(_table(world,"observable_inputs",table,check_hashes=False),1):
                if row[column] in ids: issue("DUPLICATE_PRIMARY_KEY",table,index)
                ids.add(row[column])
            registries[column]=ids
    source_hashes={}
    for table,column in (("source_obligors","obligor_id"),("source_accounts","account_id"),("source_facilities","facility_id"),("source_pools","pool_id")):
        for row in _table(world,"observable_inputs",table,check_hashes=False):
            source_hashes[row[column]]=_fingerprint({k:v for k,v in row.items() if k!="snapshot_hash"})
    for key,meta in manifest["tables"].items():
        layer,table=key.split("/"); path=_resource_path(root,meta["file"])
        if check_hashes and _sha(path)!=meta["sha256"]: issue("HASH_MISMATCH",key,None)
        n=0
        for n,r in enumerate(_table(world,layer,table,check_hashes=False),1):
            if list(r)!=meta["columns"]: issue("COLUMN_MISMATCH",key,n)
            if layer=="observable_inputs" and forbidden.intersection(r): issue("LATENT_LEAKAGE",key,n)
            if layer!="latent_truth" and r.get("known_at",r.get("decided_at",r.get("as_of",cutoff)))>cutoff: issue("FUTURE_KNOWLEDGE",key,n)
            if "observed_at" in r and "known_at" in r and r["observed_at"]>r["known_at"]: issue("KNOWLEDGE_BEFORE_OBSERVATION",key,n)
            for column,ids in registries.items():
                if column in r and r[column] is not None and r[column] not in ids and not (table=="transactions" and column=="account_id") and not (table=="source_model_versions" and column=="model_version"):
                    issue("FOREIGN_KEY",key,n)
            if "source_snapshot_hash" in r and r["source_snapshot_hash"]!=source_hashes.get(r.get("subject_id")): issue("SOURCE_SNAPSHOT_HASH",key,n)
            if "snapshot_hash" in r and r["snapshot_hash"]!=_fingerprint({k:v for k,v in r.items() if k!="snapshot_hash"}): issue("SOURCE_SNAPSHOT_HASH",key,n)
            if table=="account_months" and r["balance"]<0: issue("NEGATIVE_ACCOUNT_BALANCE",key,n)
            if table=="account_months" and not math.isclose(r["opening_balance"]+r["principal_draw"]+r["interest"]-r["payment"]-r["recovery_amount"],r["balance"],rel_tol=1e-12,abs_tol=1e-8): issue("ACCOUNT_CASHFLOW_RECONCILIATION",key,n)
            if table=="facility_exposures" and not math.isclose(r["drawn_balance"]+r["undrawn"],r["credit_limit"],rel_tol=1e-12): issue("LIMIT_RECONCILIATION",key,n)
            if table=="contractual_asset_cashflows" and not math.isclose(r["opening_principal"]-r["scheduled_principal"],r["closing_principal"],rel_tol=1e-12,abs_tol=1e-8): issue("CASHFLOW_RECONCILIATION",key,n)
            if table=="obligors": obligations.add(r["obligor_id"])
            if table=="financial_statements":
                if not math.isclose(r["assets"],r["debt"]+r["equity"],rel_tol=1e-12): issue("BALANCE_SHEET",key,n)
                if not math.isclose(r["ebitda"]-r["depreciation"],r["ebit"],rel_tol=1e-12): issue("EARNINGS",key,n)
            if table=="facilities": facilities.add(r["facility_id"])
            if table=="accounts": account_ids.add(r["account_id"])
            if table=="application_labels":
                if not r["funded"] and (r["own_loan_default"] is not None or r["account_id"] is not None): issue("REJECT_LABEL_LEAKAGE",key,n)
                if r["funded"]: funded.add(r["account_id"])
            if table=="default_events": actual_defaults+=1
            if table in ("obligor_period","application_parameters"):
                design_pd_sum+=r["true_pd"];risk_observations+=1
                latent_default_count+=int(r["potential_default_at"] is not None) if table=="obligor_period" else r["potential_loan_outcome"]
            if table=="account_events" and r["event_type"]=="DEFAULT": actual_retail_defaults.add(r["account_id"])
            if table=="pool_memberships":
                source=(r["source_world_type"],r["source_asset_id"])
                if source in sources: issue("DUPLICATE_POOL_ASSET",key,n)
                sources.add(source); pool_sums[r["pool_id"]]=pool_sums.get(r["pool_id"],0)+r["original_principal"]*r["partition_share"]
            if table=="tranches": tranche_sums[r["pool_id"]]=tranche_sums.get(r["pool_id"],0)+r["original_principal"]
            if "split" in r and "entity_split" in r:
                expected=r["entity_split"] if r["entity_split"]==r["time_split"] else "excluded"
                if r["split"]!=expected: issue("SPLIT_LEAKAGE",key,n)
        if n!=meta["rows"]: issue("ROW_COUNT",key,None)
        counts[key]=n; rows+=n
    if account_ids!=funded and manifest["world_type"]=="retail_reference_world": issue("FUNDED_ACCOUNT_LINK","accounts",None)
    for p,total in pool_sums.items():
        if not math.isclose(total,tranche_sums.get(p,0),rel_tol=1e-12,abs_tol=1e-8): issue("TRANCHE_NOTIONAL","tranches",None)
    for key,meta in manifest.get("binary_resources",{}).items():
        path=_resource_path(root,meta["file"])
        if path.stat().st_size!=math.prod(meta["shape"])*8: issue("BINARY_SHAPE",key,None)
        if check_hashes and _sha(path)!=meta["sha256"]: issue("HASH_MISMATCH",key,None)
    return _result({"passed":not issues,"issues":issues,"rows_checked":rows,"table_counts":counts,"realized_defaults":actual_defaults,"actual_funded_account_defaults":len(actual_retail_defaults),"design_vs_realized":{"risk_observations":risk_observations,"sum_conditional_pd":design_pd_sum,"latent_default_count":latent_default_count,"interpretation":"Descriptive conditional hazard sum; dependent observations do not imply a binomial test"},"diagnostic_policy":"Report realized outcomes without reseeding or target-rate adjustment","data_origin":"SYNTHETIC"},"synthetic.validate_synthetic_world","synthetic_validation",context,provenance={"data_origin":"SYNTHETIC","parameter_origin":"SYNTHETIC_REFERENCE"})
