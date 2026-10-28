"""Paired model comparisons and dated local rating benchmarks."""
from copy import deepcopy
import numpy as np
from .core import _fail,_fingerprint
from .data import _date
from ._expert_common import _ex_object,_ex_ids,_ex_target
from ._monitoring_metrics import _cohort,_metrics
from ._validation_common import _vb_design,_vb_draws,_vb_bands,_vb_result,_vb_string


def benchmark_predictions(model_cohorts,*,reference_model_id,metrics,matching_policy,multiple_comparisons,design,uncertainty,cluster_resamples=None,confidence_level=.95,context=None):
    """Compare frozen models on the same mature sealed monitoring-cohort cases."""
    d=_vb_design(design,context);entries=_ex_object(model_cohorts,'model_cohorts');models=_ex_ids(sorted(entries),'model IDs');requested=_ex_ids(metrics,'metrics')
    if len(models)<2 or reference_model_id not in models:_fail('BENCHMARK_MODELS','At least two models including the declared reference required')
    if set(requested)-{'brier','log_loss','auc','gini','calibration_error','mean_pd'}:_fail('BENCHMARK_METRIC','Unknown paired metric')
    if matching_policy not in ('strict','intersection')or multiple_comparisons not in ('none','bonferroni'):_fail('BENCHMARK_POLICY','Declare matching and multiple-comparison interval policy')
    if d['target']['conditioning']!='performing'or d['target']['time_unit']!='days':_fail('BENCHMARK_TARGET','Monitoring cohort comparisons require a performing day-horizon default target')
    bundles={};maps={};manifests={};target=None
    for model_id in models:
        entry=_ex_object(entries[model_id],'model cohort entry');bundle=deepcopy(_cohort(entry.get('cohort')))
        for key in ('model_version','parameter_version','model_fingerprint'):_vb_string(entry.get(key),key)
        known=_date(entry.get('training_known_at'),'training_known_at');rows=bundle['rows'];_ex_ids([r.get('case_id')for r in rows],'unique cohort case IDs',empty=True)
        if any(r['model_id']!=model_id or r['model_version']!=entry['model_version']for r in rows):_fail('BENCHMARK_VERSION','Cohort rows differ from model identity/version')
        if any(known>_date(r['forecast_at'],'forecast_at')for r in rows):_fail('BENCHMARK_LOOKAHEAD','Model unavailable at historical forecast time')
        if any(r.get('prediction_origin')not in (None,'HOLDOUT','EXTERNAL')for r in rows):_fail('BENCHMARK_DESIGN','In-sample or model-generated outcome cohorts cannot claim held-out comparison')
        t=bundle['target_contract']
        if bundle['evaluation_cutoff']!=d['evaluation_cutoff']or t.get('event')!='default'or t.get('probability_measure')!='PHYSICAL'or t.get('horizon_days')!=d['target']['horizon']:_fail('BENCHMARK_TARGET','Cohort horizon, default target and cutoff must match design')
        if target is not None and _fingerprint(target)!=_fingerprint(t):_fail('BENCHMARK_TARGET','All cohort definitions must match exactly')
        target=t
        for row in rows:
            if row['eligible']:
                if _date(row['outcome_end'],'outcome_end')>_date(d['evaluation_cutoff'],'evaluation_cutoff')or _date(row['outcome_known_at'],'outcome_known_at')>_date(d['evaluation_cutoff'],'evaluation_cutoff'):_fail('BENCHMARK_MATURITY','Eligible labels must be mature and known')
                if row['target']==0 and(row.get('complete_through')is None or _date(row['complete_through'],'complete_through')<_date(row['outcome_end'],'outcome_end')):_fail('BENCHMARK_COVERAGE','Negative labels require complete-through coverage')
        bundles[model_id]=bundle;maps[model_id]={r['case_id']:r for r in rows};manifests[model_id]={k:deepcopy(entry[k])for k in ('model_version','parameter_version','model_fingerprint','training_known_at')};manifests[model_id]['cohort_hash']=bundle['cohort_hash']
    universe=sorted(set().union(*(set(m)for m in maps.values())))
    if matching_policy=='strict'and any(set(m)!=set(universe)for m in maps.values()):_fail('BENCHMARK_MATCHING','Strict comparison requires identical case IDs in every model')
    comparison_rows=[];exclusions=[];paired=set();missing={model_id:sorted(set(universe)-set(maps[model_id]))for model_id in models}
    for case_id in universe:
        present=[maps[m][case_id]for m in models if case_id in maps[m]];row=deepcopy(maps[reference_model_id].get(case_id,present[0]));comparison_rows.append(row);reason=None
        if len(present)!=len(models):reason='MISSING_MODEL_PREDICTION'
        else:
            for key in ('entity_id','cluster_id','forecast_at','outcome_end','weight'):
                if any(r[key]!=present[0][key]for r in present):_fail('BENCHMARK_ALIGNMENT','Matched case identities, dates and evaluation weights differ')
            labels={r['target']for r in present if r['target']is not None}
            if len(labels)>1:_fail('BENCHMARK_OUTCOME','Models must be evaluated against the same observed outcome')
            if not all(r['eligible']for r in present):reason='OUTCOME_NOT_ELIGIBLE_IN_EVERY_COHORT'
            elif present[0]['weight']==0:reason='ZERO_EVALUATION_WEIGHT'
            else:paired.add(case_id)
        if reason is not None:exclusions.append(dict(prediction_id=case_id,reason=reason))
    for model_id in models:
        for row in maps[model_id].values():row['eligible']=row['case_id']in paired
    draws=_vb_draws(uncertainty,cluster_resamples,comparison_rows,confidence_level)
    def calculate(multiplicities=None):
        scores={m:_metrics([maps[m][k]for k in sorted(maps[m])],multiplicities)for m in models};comparisons={}
        for m in models:
            if m==reference_model_id:continue
            comparisons[m]={}
            for metric in requested:
                a=scores[m][metric];b=scores[reference_model_id][metric];good=a is not None and b is not None
                reason=None if good else'INFINITE_OR_UNIDENTIFIED_LOG_LOSS'if metric=='log_loss'else'NON_ESTIMABLE_PAIRED_METRIC'
                comparisons[m][metric]=dict(difference=a-b if good else None,candidate=a,reference=b,applicability='APPLICABLE'if good else'NOT_APPLICABLE',reason=reason)
        return scores,comparisons
    scores,comparisons=calculate();samples=None
    if draws is not None:
        samples=[]
        for draw in draws:
            multiplicities={r['cluster_id']:float(v)for r,v in zip(comparison_rows,draw)};_,c=calculate(multiplicities);samples.append({m+'/'+metric:c[m][metric]['difference']for m in c for metric in requested})
    family_size=(len(models)-1)*len(requested);marginal_level=1-(1-confidence_level)/family_size if multiple_comparisons=='bonferroni'else confidence_level
    bands=_vb_bands(samples,marginal_level);bands.update(multiple_comparisons=multiple_comparisons,family_size=family_size,family_confidence_level=confidence_level,coverage_claim='BONFERRONI_COMBINATION_OF_APPROXIMATE_MARGINAL_BOOTSTRAP_INTERVALS'if multiple_comparisons=='bonferroni'else'NO_FAMILYWISE_COVERAGE_ADJUSTMENT')
    diagnostics=dict(reference_model_id=reference_model_id,models=manifests,cohort_target_contract=target,matching_policy=matching_policy,missing_cases_by_model=missing,paired_case_ids=sorted(paired),difference_direction='CANDIDATE_MINUS_REFERENCE',interpretation=dict(brier='LOWER_BETTER',log_loss='LOWER_BETTER',auc='HIGHER_BETTER',gini='HIGHER_BETTER',calibration_error='SIGNED_LEVEL_DIFFERENCE_NOT_AUTOMATIC_IMPROVEMENT',mean_pd='LEVEL_DIFFERENCE_NOT_AUTOMATIC_IMPROVEMENT'),comparison_scope='COMMON_MATURE_OBSERVED_CASES_ONLY',outcome_dependent_candidate_selection_not_corrected=True)
    return _vb_result('benchmark_predictions',dict(by_model=scores,paired_differences=comparisons),d,comparison_rows,exclusions,bands,context,diagnostics,'paired_model_comparison')


def benchmark_ratings(internal_ratings,benchmark_records,*,scale_mapping,matching_policy,design,uncertainty,cluster_resamples=None,confidence_level=.95,context=None):
    """Match dated local rating histories and compare explicit monotone grade mappings."""
    from .expert import assess_expert_rating_agreement
    from ._validation_common import _vb_records,_vb_number
    d=_vb_design(design,context,False);rows=sorted(_vb_records(internal_ratings,d),key=lambda r:r['prediction_id']);mapping=deepcopy(_ex_object(scale_mapping,'scale_mapping'));policy=_ex_object(matching_policy,'matching_policy')
    for key in ('mapping_id','version','reference'):_vb_string(mapping.get(key),key)
    known=_date(mapping.get('known_at'),'mapping known_at')
    if known>_date(d['evaluation_cutoff'],'evaluation_cutoff')or any(known>_date(r['forecast_known_at'],'forecast_known_at')for r in rows):_fail('BENCHMARK_LOOKAHEAD','Scale mapping must be frozen before comparisons')
    grades=_ex_ids(mapping.get('common_grades'),'common_grades');rank={g:i for i,g in enumerate(grades)}
    if len(grades)<2:_fail('BENCHMARK_SCALE','At least two ordered common grades required')
    for side in ('internal','benchmark'):
        spec=_ex_object(mapping.get(side),side+' scale')
        for key in ('scale_id','scale_version'):_vb_string(spec.get(key),key)
        source=_ex_ids(spec.get('grades'),side+' grades');conversion=_ex_object(mapping.get(side+'_to_common'),side+' mapping')
        if set(conversion)!=set(source)or any(v not in rank for v in conversion.values())or np.any(np.diff([rank[conversion[g]]for g in source])<0):_fail('BENCHMARK_SCALE','Complete order-preserving mappings to best-to-worst common grades required')
    if policy.get('benchmark_origin')not in ('EXTERNAL_JUDGEMENT','INTERNAL_DERIVED_MAPPING','INTERNAL_CHALLENGER'):_fail('BENCHMARK_POLICY','Declare benchmark origin; internally derived mappings are not independent evidence')
    if policy.get('method')not in ('exact_date','latest_known')or policy.get('watch_policy')not in ('include','exclude')or policy.get('kappa_weighting')not in ('unweighted','linear','quadratic'):_fail('BENCHMARK_POLICY','Declare date, watch and kappa-weighting policies')
    provider=_vb_string(policy.get('provider_id'),'provider_id');age=_vb_number(policy.get('max_age_days'),'max_age_days',0)
    if age!=int(age):_fail('BENCHMARK_POLICY','Staleness limit must be integer days')
    if not isinstance(benchmark_records,(list,tuple))or any(not isinstance(r,dict)for r in benchmark_records):_fail('BENCHMARK_RECORDS','Local benchmark record array required')
    external=sorted(deepcopy(list(benchmark_records)),key=lambda r:_vb_string(r.get('benchmark_id'),'benchmark_id'));_ex_ids([r['benchmark_id']for r in external],'benchmark IDs',empty=True);dimensions=('rating_type','support_concept','currency','rating_philosophy')
    external_index={}
    for row in external:
        _ex_target(row.get('target'))
        for key in ('entity_id','rating_unit_id','provider_id',*dimensions):_vb_string(row.get(key),key)
        if row.get('status')not in ('ACTIVE','WATCH','WITHDRAWN','MISSING'):_fail('BENCHMARK_STATUS','Unknown benchmark status')
        if _date(row.get('known_at'),'benchmark known_at')<_date(row.get('rating_at'),'rating_at'):_fail('BENCHMARK_LOOKAHEAD','Benchmark cannot be known before its effective rating date')
        external_index.setdefault((row['entity_id'],row['rating_unit_id'],row['provider_id']),[]).append(row)
    n=len(rows);left=np.full(n,-1,dtype=int);right=np.full(n,-1,dtype=int);exclusions=[];matches=[];audit=[]
    for i,row in enumerate(rows):
        for key in ('rating_unit_id',*dimensions):_vb_string(row.get(key),key)
        if row.get('scale_id')!=mapping['internal']['scale_id']or row.get('scale_version')!=mapping['internal']['scale_version']:_fail('BENCHMARK_SCALE','Internal scale/version differs from frozen mapping')
        status=row.get('rating_status')
        if status not in ('ACTIVE','WATCH','WITHDRAWN','MISSING'):_fail('BENCHMARK_STATUS','Declare internal rating status')
        reason=None;selected=None
        if _date(row['as_of'],'as_of')>_date(d['evaluation_cutoff'],'evaluation_cutoff'):reason='FUTURE_INTERNAL_RATING'
        elif status in ('WITHDRAWN','MISSING')or row.get('grade')is None:reason='NO_ACTIVE_INTERNAL_RATING'
        elif status=='WATCH'and policy['watch_policy']=='exclude':reason='INTERNAL_WATCH_EXCLUDED'
        elif row['grade']not in mapping['internal_to_common']:_fail('BENCHMARK_SCALE','Unknown internal grade')
        identity=external_index.get((row['entity_id'],row['rating_unit_id'],provider),[])
        available=[r for r in identity if _date(r['known_at'],'known_at')<=_date(row['forecast_known_at'],'forecast_known_at')and _date(r['rating_at'],'rating_at')<=_date(row['as_of'],'as_of')]
        concept=[r for r in available if all(r[k]==row[k]for k in dimensions)and r.get('target')==d['target']]
        candidates=[r for r in concept if policy['method']=='latest_known'or r['rating_at']==row['as_of']]
        if reason is None:
            if not identity:reason='NO_IDENTITY_MATCH'
            elif not available:reason='NO_KNOWN_BENCHMARK'
            elif not concept:reason='NO_MATCHING_RATING_CONCEPT'
            elif not candidates:reason='NO_MATCHING_RATING_DATE'
            else:
                latest=max((r['rating_at'],r['known_at'])for r in candidates);tied=[r for r in candidates if(r['rating_at'],r['known_at'])==latest]
                if len(tied)!=1:_fail('BENCHMARK_AMBIGUITY','Multiple equally current benchmark records require explicit upstream resolution')
                selected=tied[0];elapsed=(_date(row['as_of'],'as_of')-_date(selected['rating_at'],'rating_at')).days
                if elapsed>age:reason='STALE_BENCHMARK'
                elif selected.get('scale_id')!=mapping['benchmark']['scale_id']or selected.get('scale_version')!=mapping['benchmark']['scale_version']:reason='BENCHMARK_SCALE_VERSION_MISMATCH'
                elif selected['status']=='WITHDRAWN':reason='WITHDRAWN_BENCHMARK'
                elif selected['status']=='MISSING'or selected.get('grade')is None:reason='MISSING_BENCHMARK_GRADE'
                elif selected['status']=='WATCH'and policy['watch_policy']=='exclude':reason='BENCHMARK_WATCH_EXCLUDED'
                elif selected['grade']not in mapping['benchmark_to_common']:_fail('BENCHMARK_SCALE','Unknown selected benchmark grade')
                else:
                    a=mapping['internal_to_common'][row['grade']];b=mapping['benchmark_to_common'][selected['grade']];left[i]=rank[a];right[i]=rank[b]
                    matches.append(dict(prediction_id=row['prediction_id'],benchmark_id=selected['benchmark_id'],entity_id=row['entity_id'],rating_unit_id=row['rating_unit_id'],internal_grade=row['grade'],benchmark_grade=selected['grade'],internal_common_grade=a,benchmark_common_grade=b,signed_notch_difference=int(left[i]-right[i]),benchmark_age_days=elapsed,benchmark_known_at=selected['known_at']))
        audit.append(dict(prediction_id=row['prediction_id'],identity_candidates=len(identity),known_candidates=len(available),concept_candidates=len(concept),date_candidates=len(candidates),selected_benchmark_id=None if selected is None else selected['benchmark_id'],status='MATCHED'if reason is None else reason))
        if reason is not None:exclusions.append(dict(prediction_id=row['prediction_id'],reason=reason))
    active=left>=0;point=assess_expert_rating_agreement([grades[x]for x in left[active]],[grades[x]for x in right[active]],ordered_grades=grades,rater_pair=['internal','benchmark'],weighting=policy['kappa_weighting'],missing_policy='error').value;agreement_weights=np.array(point['agreement_weights']);draws=_vb_draws(uncertainty,cluster_resamples,rows,confidence_level);k=len(grades)
    def calculate(draw):
        table=np.zeros((k,k));weight=draw*active;valid=weight>0;np.add.at(table,(left[valid],right[valid]),weight[valid]);N=float(table.sum());empty=N==0
        result=dict(matched_weight=N,confusion_matrix=table.tolist(),exact_agreement=None,within_one_notch=None,mean_signed_notches=None,mean_absolute_notches=None,kappa=None,spearman_rank_correlation=None,observed_weighted_agreement=None,expected_weighted_agreement=None,kappa_applicability='NOT_APPLICABLE',rank_applicability='NOT_APPLICABLE')
        if empty:return result
        joint=table/N;difference=np.arange(k)[:,None]-np.arange(k)[None,:];a=joint.sum(axis=1);b=joint.sum(axis=0);observed=float(np.sum(joint*agreement_weights));expected=float(np.sum(np.outer(a,b)*agreement_weights));u=np.cumsum(a)-a/2;v=np.cumsum(b)-b/2;u=u-a@u;v=v-b@v;variance=(a@(u*u))*(b@(v*v));correlation=float(u@joint@v/np.sqrt(variance))if variance>0 else None
        result.update(exact_agreement=float(np.trace(joint)),within_one_notch=float(np.sum(joint[np.abs(difference)<=1])),mean_signed_notches=float(np.sum(joint*difference)),mean_absolute_notches=float(np.sum(joint*np.abs(difference))),kappa=None if expected>=1-1e-15 else(observed-expected)/(1-expected),spearman_rank_correlation=correlation,observed_weighted_agreement=observed,expected_weighted_agreement=expected,kappa_applicability='NOT_APPLICABLE'if expected>=1-1e-15 else'APPLICABLE',rank_applicability='NOT_APPLICABLE'if correlation is None else'APPLICABLE')
        return result
    metrics=calculate(np.ones(n));keys=('exact_agreement','within_one_notch','mean_signed_notches','mean_absolute_notches','kappa','spearman_rank_correlation');samples=None if draws is None else[{key:value for key,value in calculate(draw).items()if key in keys}for draw in draws]
    diagnostics=dict(scale_mapping=mapping,matching_policy=policy,matches=matches,matching_audit=audit,benchmark_fingerprint=_fingerprint(external),ordered_common_grades=grades,notch_direction='POSITIVE_INTERNAL_WORSE_THAN_BENCHMARK',benchmark_is_ground_truth=False,comparison_type='INTERNAL_MAPPING_CHECK'if policy['benchmark_origin']=='INTERNAL_DERIVED_MAPPING'else policy['benchmark_origin']+'_AGREEMENT',independent_outcome_calibration_evidence=False,no_fallback_past_known_withdrawal=True,benchmark_count=len(external),distinct_benchmarks_used=len({m['benchmark_id']for m in matches}),unused_benchmark_ids=sorted({r['benchmark_id']for r in external}-{m['benchmark_id']for m in matches}))
    return _vb_result('benchmark_ratings',metrics,d,rows,exclusions,_vb_bands(samples,confidence_level),context,diagnostics,'ordinal_rating_agreement')
