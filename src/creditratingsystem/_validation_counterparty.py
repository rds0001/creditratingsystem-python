"""Per-netting-set exposure validation and separate observed-default closeouts."""
from copy import deepcopy
import numpy as np
from .core import _fail, _fingerprint, _numeric
from .data import _date
from .ead import _path_artifact, closeout_exposure_at_default
from ._expert_common import _ex_object, _ex_ids
from ._validation_common import _vb_design, _vb_records, _vb_number, _vb_string, _vb_errors, _vb_draws, _vb_bands, _vb_result
from ._validation_retail import _vr_rows


def counterparty_exposure_backtest(forecasts, observations, *, quantile_levels, design, uncertainty,
                                  cluster_resamples=None, confidence_level=.95, context=None):
    """Score physical exposure distributions with frozen legal/CSA reconciliation."""
    d = _vb_design(design, context); currency = _vb_string(d.get('currency'), 'design currency')
    rows = _vr_rows(forecasts, 'forecasts'); observed = _vr_rows(observations, 'observations')
    for row in rows + observed:
        for key in ('netting_paths', 'collateral_paths'):
            if hasattr(row.get(key), 'value'): row[key] = row[key].value
    rows = _vb_records(rows, d)
    _ex_ids([r.get('prediction_id') for r in observed], 'observation prediction IDs', empty=True)
    obs = {r['prediction_id']: r for r in observed}
    if not set(obs) <= {r['prediction_id'] for r in rows}:
        _fail('CCR_VALIDATION_ALIGNMENT', 'Observation references unknown forecast')
    q = _numeric(quantile_levels, 'quantile_levels')
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or np.any((q <= 0) | (q >= 1)) or len(set(q)) != len(q):
        _fail('CCR_VALIDATION_QUANTILES', 'Distinct interior quantile levels required')
    q = np.sort(q); n = len(rows); p = np.zeros(n); y = np.zeros(n); cp = np.zeros(n); cy = np.zeros(n)
    weights = np.ones(n); eligible = np.zeros(n, bool); celigible = np.zeros(n, bool)
    thresholds = np.zeros((n,len(q))); tails = np.zeros_like(thresholds)
    exclusions, traces, violations, closeout_traces = [], [], [], []
    cutoff = _date(d['evaluation_cutoff'], 'evaluation cutoff')
    def issue(identity, code, detail): violations.append(dict(prediction_id=identity, code=code, detail=detail))
    def agreement(artifact, row):
        found = [a for a in artifact['netting_agreements'] if a['netting_set_id'] == row['netting_set_id']]
        if len(found) != 1 or found[0]['counterparty_id'] != row['counterparty_id']:
            _fail('CCR_VALIDATION_SCOPE', 'Counterparty/netting-set must match exact frozen agreement')
        return found[0]
    def artifacts(row, forecast):
        net = _path_artifact(row.get('netting_paths'), 'NETTING_PATHS')
        if net['as_of'] != forecast['as_of'] or net['currency'] != currency:
            _fail('CCR_VALIDATION_ARTIFACT', 'Path snapshot/currency differs from forecast origin')
        agreement(net, forecast)
        coll = None
        if forecast['exposure_target'] == 'COLLATERALIZED':
            coll = _path_artifact(row.get('collateral_paths'), 'COLLATERAL_PATHS')
            if coll['netting_artifact_hash'] != net['artifact_hash']:
                _fail('CCR_VALIDATION_BINDING', 'Collateral artifact does not bind matching netting paths')
        elif row.get('collateral_paths') is not None:
            _fail('CCR_VALIDATION_TARGET', 'Uncollateralized target cannot silently include collateral')
        return net, coll
    for i,row in enumerate(rows):
        identity = row['prediction_id']; weights[i] = _vb_number(row.get('weight',1), 'evaluation weight',0)
        for key in ('counterparty_id','netting_set_id','legal_version'):
            _vb_string(row.get(key),key)
        if row.get('exposure_target') not in ('UNCOLLATERALIZED','COLLATERALIZED'):
            _fail('CCR_VALIDATION_TARGET','Explicit uncollateralized/collateralized exposure target required')
        if row['exposure_target']=='COLLATERALIZED': _vb_string(row.get('csa_version'),'csa_version')
        net, coll = artifacts(row,row); ag = agreement(net,row)
        if ag['known_at'] > row['forecast_known_at']:
            _fail('CCR_VALIDATION_LOOKAHEAD','Netting agreement unavailable at forecast')
        if coll is not None and any(a['known_at'] > row['forecast_known_at'] for a in coll['margin_policies']):
            _fail('CCR_VALIDATION_LOOKAHEAD','CSA policy unavailable at forecast')
        path_ids = list(net['path_ids']); pw = row.get('path_weights')
        if pw is None: w = np.ones(len(path_ids))/len(path_ids)
        else:
            pw = _ex_object(pw,'path_weights')
            if set(pw)!=set(path_ids): _fail('CCR_VALIDATION_PATHS','Path weights must cover exact frozen scenario IDs')
            w=np.array([_vb_number(pw[k],'path weight',0)for k in path_ids])
            if not np.any(w>0): _fail('CCR_VALIDATION_PATHS','Positive scenario weight required')
            w=w/np.max(w);w=w/w.sum()
        source = coll if coll is not None else net; field = 'residual_exposure'if coll is not None else'positive_exposure'
        selected = {r['path_id']:r for r in source['paths']if r['netting_set_id']==row['netting_set_id']and r['date']==row['horizon_end']}
        if set(selected)!=set(path_ids): _fail('CCR_VALIDATION_HORIZON','Exact common scenario/horizon/set path slice required')
        x=np.array([_vb_number(selected[k][field],'exposure sample',0)for k in path_ids]);p[i]=float(w@x)
        order=np.argsort(x,kind='stable');cdf=np.cumsum(w[order]);cdf[-1]=1.
        for j,level in enumerate(q):
            thresholds[i,j]=x[order[min(np.searchsorted(cdf,level,side='left'),len(x)-1)]]
            tails[i,j]=float(w@(x>thresholds[i,j]))
        actual=obs.get(identity);reason=None;recomputed=None;on=oc=None;valid_contract=True
        if _date(row['horizon_end'],'horizon_end')>cutoff:reason='IMMATURE_HORIZON'
        elif actual is None or actual.get('observation_status')=='CENSORED':reason='EXPOSURE_NOT_OBSERVED'
        else:
            if actual.get('observation_status')!='OBSERVED':_fail('CCR_VALIDATION_OBSERVATION','Explicit OBSERVED or CENSORED status required')
            at=_date(actual.get('observed_at'),'observed_at');known=_date(actual.get('known_at'),'observation known_at')
            if known<at or actual['observed_at']!=row['horizon_end']:_fail('CCR_VALIDATION_ALIGNMENT','Exact endpoint observation with subsequent knowledge required')
            if known>cutoff:reason='EXPOSURE_NOT_YET_KNOWN'
            else:
                on,oc=artifacts(actual,row)
                if _fingerprint(on['netting_agreements'])!=_fingerprint(net['netting_agreements']) or actual.get('legal_version')!=row['legal_version']:
                    issue(identity,'NETTING_CONTRACT_CHANGED','Observed legal membership/version differs from frozen forecast');valid_contract=False
                if coll is not None and (_fingerprint(oc['margin_policies'])!=_fingerprint(coll['margin_policies'])or actual.get('csa_version')!=row['csa_version']):
                    issue(identity,'CSA_CONTRACT_CHANGED','Observed margin/posted-claim contract/version differs from frozen forecast');valid_contract=False
                os=oc if oc is not None else on
                candidates=[r for r in os['paths']if r['path_id']==actual.get('observed_path_id')and r['date']==row['horizon_end']and r['netting_set_id']==row['netting_set_id']]
                if len(candidates)!=1:_fail('CCR_VALIDATION_ALIGNMENT','Unique actual path/set/endpoint required')
                recomputed=_vb_number(candidates[0][field],'observed exposure',0);reported=_vb_number(actual.get('reported_exposure'),'reported exposure',0)
                if not np.isclose(reported,recomputed,rtol=1e-10,atol=1e-12):
                    issue(identity,'ILLEGAL_OFFSET_OR_EXPOSURE_RECONCILIATION','Reported exposure differs from exact per-set collateral/netting exposure');valid_contract=False
                y[i]=recomputed
                if not valid_contract:reason='CONTRACT_OR_EXPOSURE_RECONCILIATION'
                elif weights[i]==0:reason='ZERO_EVALUATION_WEIGHT'
                else:eligible[i]=True
        if reason:exclusions.append(dict(prediction_id=identity,reason=reason))
        trace=dict(prediction_id=identity,counterparty_id=row['counterparty_id'],netting_set_id=row['netting_set_id'],forecast_netting_hash=net['artifact_hash'],forecast_collateral_hash=None if coll is None else coll['artifact_hash'],forecast_mean=float(p[i]),quantiles=[dict(level=float(level),value=float(thresholds[i,j]),model_strict_tail_probability=float(tails[i,j]))for j,level in enumerate(q)],recomputed_observed_exposure=recomputed,eligible=bool(eligible[i]))
        traces.append(trace)
        # Default-conditioned closeout is a separate subset, never an all-observation loss label.
        ctr=dict(prediction_id=identity,applicability='NOT_APPLICABLE',reason='NO_OBSERVED_DEFAULT_CLOSEOUT',predicted_mean=None,observed_exposure=None,simulated_default_paths=0)
        forecast_events=row.get('default_events',[])
        if forecast_events:
            sampling=_ex_object(row.get('default_sampling_contract'),'default_sampling_contract')
            _vb_string(sampling.get('reference'),'default sampling reference')
            if sampling.get('mode')!='JOINT_SIMULATED_PATHS' or _date(sampling.get('known_at'),'default sampling known_at')>_date(row['forecast_known_at'],'forecast_known_at'):_fail('CCR_VALIDATION_DEFAULT_SAMPLING','Historically frozen joint simulated default-path contract required')
            if coll is None:_fail('CCR_VALIDATION_CLOSEOUT','Closeout validation requires explicit collateral artifact, including zero-collateral policy')
            simulated=closeout_exposure_at_default(net,coll,forecast_events,row.get('intermediate_cashflows',[]),freeze_timing=row.get('freeze_timing'),cashflow_convention=row.get('cashflow_convention')).value['closeouts']
            simulated=[r for r in simulated if r['netting_set_id']==row['netting_set_id']and row['as_of']<=r['default_date']<=row['horizon_end']]
            sw=np.array([w[path_ids.index(r['path_id'])]for r in simulated]);sx=np.array([r['closeout_exposure']for r in simulated]);ctr['simulated_default_paths']=len(simulated)
            if len(sw)and sw.sum()>0:cp[i]=float((sw/sw.sum())@sx);ctr['predicted_mean']=float(cp[i])
            if actual is not None and actual.get('closeout_event')is not None:
                event=actual['closeout_event'];end=_date(event.get('closeout_date'),'closeout_date');available=_date(actual.get('closeout_known_at'),'closeout_known_at')
                if available<end or event.get('path_id')!=actual.get('observed_path_id')or event.get('netting_set_id')!=row['netting_set_id']or not row['as_of']<=event.get('default_date','')<=row['horizon_end']:_fail('CCR_VALIDATION_CLOSEOUT','Aligned dated observed default and closeout knowledge required')
                if end>cutoff or available>cutoff or _date(row['horizon_end'],'horizon_end')>cutoff:ctr['reason']='IMMATURE_OR_UNKNOWN_DEFAULT_CLOSEOUT'
                elif not valid_contract:ctr['reason']='UNVERIFIED_OBSERVED_CONTRACT'
                elif not len(sw)or sw.sum()==0:ctr['reason']='NO_SIMULATED_DEFAULT_SUPPORT'
                else:
                    if on is None:
                        on,oc=artifacts(actual,row)
                        valid_contract=_fingerprint(on['netting_agreements'])==_fingerprint(net['netting_agreements']) and actual.get('legal_version')==row['legal_version'] and _fingerprint(oc['margin_policies'])==_fingerprint(coll['margin_policies']) and actual.get('csa_version')==row['csa_version']
                        if not valid_contract:
                            issue(identity,'CLOSEOUT_CONTRACT_CHANGED','Observed default closeout legal/CSA contract differs from frozen forecast')
                            ctr['reason']='UNVERIFIED_OBSERVED_CONTRACT';closeout_traces.append(ctr);continue
                    closed=closeout_exposure_at_default(on,oc,[event],actual.get('intermediate_cashflows',[]),freeze_timing=row.get('freeze_timing'),cashflow_convention=row.get('cashflow_convention')).value['closeouts'][0]
                    cy[i]=closed['closeout_exposure'];reported=_vb_number(actual.get('reported_closeout_exposure'),'reported closeout exposure',0)
                    if not np.isclose(reported,cy[i],rtol=1e-10,atol=1e-12):issue(identity,'CLOSEOUT_RECONCILIATION','Reported closeout differs from frozen-units/date-aware calculation');ctr['reason']='CLOSEOUT_RECONCILIATION'
                    elif weights[i]==0:ctr['reason']='ZERO_EVALUATION_WEIGHT'
                    else:celigible[i]=True;ctr.update(applicability='APPLICABLE',reason=None,observed_exposure=float(cy[i]),closeout_details=closed)
        elif actual is not None and actual.get('closeout_event')is not None:ctr['reason']='NO_FROZEN_DEFAULT_FORECAST'
        closeout_traces.append(ctr)
    def calculate(draw):
        ew=weights*draw*eligible;total=ew.sum();quant=[]
        for j,level in enumerate(q):
            pinball=np.where(y>=thresholds[:,j],level*(y-thresholds[:,j]),(1-level)*(thresholds[:,j]-y))
            quant.append(dict(level=float(level),exception_rate=None if total==0 else float(ew@(y>thresholds[:,j])/total),nominal_tail_probability=float(1-level),mean_model_strict_tail_probability=None if total==0 else float(ew@tails[:,j]/total),pinball_loss=None if total==0 else float(ew@pinball/total),applicability='APPLICABLE'if total>0 else'NOT_APPLICABLE'))
        return dict(market_exposure=_vb_errors(p,y,ew),quantile_exceptions=quant,conditional_default_closeout=_vb_errors(cp,cy,weights*draw*celigible))
    metrics=calculate(np.ones(n));draws=_vb_draws(uncertainty,cluster_resamples,rows,confidence_level);samples=None
    if draws is not None:
        samples=[]
        for draw in draws:
            m=calculate(draw);sample={kind+'/'+metric:m[kind][metric]for kind in ('market_exposure','conditional_default_closeout')for metric in ('bias','rmse')}
            for j,v in enumerate(m['quantile_exceptions']):
                for key in ('exception_rate','pinball_loss'):sample['quantile_'+str(j)+'/'+key]=v[key]
            samples.append(sample)
    diagnostics=dict(forecast_traces=traces,closeout_traces=closeout_traces,reconciliation_violations=violations,observation_fingerprint=_fingerprint(observed),quantile_convention='WEIGHTED_EMPIRICAL_INVERSE_CDF_LEFT',exception_convention='STRICTLY_GREATER_THAN_QUANTILE',cross_set_offset_permitted=False,rare_default_loss_assessed=False,closeout_estimand='EXPOSURE_CONDITIONAL_ON_SIMULATED_DEFAULT_WITHIN_FORECAST_HORIZON',path_sampling_uncertainty_assessed=False)
    result=_vb_result('counterparty_exposure_backtest',metrics,d,rows,exclusions,_vb_bands(samples,confidence_level),context,diagnostics,currency)
    result.value['diagnostics']['closeout_included_count']=int(celigible.sum())
    if np.any(celigible):result.value['applicability']='APPLICABLE';result.value['reason']=None
    if violations:result.value['decision']=dict(status='FAIL',rule='FROZEN_CONTRACT_AND_EXPOSURE_RECONCILIATION')
    return result
