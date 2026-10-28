"""Training-only feature transformations and auditable feature construction."""
from collections.abc import Mapping
from datetime import date,timedelta
import calendar
import math
import numpy as np
from .core import _fail,_numeric,_model_artifact,_fit_result,_result,_fingerprint


def _missing(v):
    return v is None or isinstance(v,(float,np.floating)) and math.isnan(v)


def _records(rows):
    if not isinstance(rows,(list,tuple)) or not rows or any(not isinstance(r,Mapping) for r in rows):
        _fail('INVALID_TABLE','Nonempty list of row objects required')
    return rows


def _number(v,name):
    a=_numeric(v,name)
    if a.ndim: _fail('INVALID_INPUT',f'{name} must be scalar')
    return float(a)


def _artifact(method,features,parameters,target,training,model_id,model_version):
    if not isinstance(training,Mapping):_fail('INVALID_MODEL','Explicit training metadata required')
    return _model_artifact(dict(schema_version='1.0',model_id=model_id,model_version=model_version,method_id=method,
        feature_order=features,parameters=parameters,preprocessing={'transformation':'frozen'},target=target,training=training,
        diagnostics={},provenance={'data_origin':training.get('data_origin','USER_SUPPLIED'),'parameter_origin':'ESTIMATED'}))


def fit_preprocessor(train_features,feature_schema,*,target,training,model_id,model_version,context=None):
    """Freeze numeric imputers/scalers and categorical vocabularies from training rows."""
    rows=_records(train_features)
    if not isinstance(feature_schema,(list,tuple)) or not feature_schema: _fail('INVALID_SCHEMA','Ordered feature specifications required')
    names=[];specs=[];output=[];lineage=[]
    for spec in feature_schema:
        if not isinstance(spec,Mapping) or not isinstance(spec.get('name'),str) or not spec['name'] or spec['name'] in names:
            _fail('INVALID_SCHEMA','Unique feature names required')
        name=spec['name'];names.append(name)
        if any(name not in row for row in rows): _fail('MISSING_FEATURE',f'{name} missing from training schema')
        values=[row[name] for row in rows];observed=[v for v in values if not _missing(v)];missing=len(values)-len(observed)
        fitted={'name':name,'type':spec.get('type'),'missing_count':missing,'n_training':len(rows)}
        if spec.get('type')=='numeric':
            policy=spec.get('imputation');scale=spec.get('scaling')
            if policy not in ('error','mean','median','constant') or scale not in ('none','standard','robust'):
                _fail('INVALID_SCHEMA','Declare numeric imputation and scaling policies')
            numeric=np.array([_number(v,name) for v in observed])
            if policy=='error' and missing: _fail('MISSING_INPUT',f'{name} training values missing')
            if policy in ('mean','median') and len(numeric)==0: _fail('NO_OBSERVED_VALUES',f'{name} cannot estimate an imputer')
            fill=_number(spec.get('fill_value'),'fill_value') if policy=='constant' else float(np.mean(numeric)) if policy=='mean' else float(np.median(numeric)) if policy=='median' else None
            complete=np.array([fill if _missing(v) else _number(v,name) for v in values])
            center=float(np.mean(complete)) if scale=='standard' else float(np.median(complete)) if scale=='robust' else 0.
            divisor=float(np.std(complete,ddof=0)) if scale=='standard' else float(np.quantile(complete,.75)-np.quantile(complete,.25)) if scale=='robust' else 1.
            constant=divisor==0
            if constant:divisor=1.
            indicator=spec.get('missing_indicator',False)
            if not isinstance(indicator,bool):_fail('INVALID_SCHEMA','missing_indicator must be boolean')
            fitted.update(imputation=policy,fill_value=fill,scaling=scale,center=center,divisor=divisor,constant_training_scale=constant,missing_indicator=indicator)
            columns=[name]+([name+'::missing'] if indicator else [])
        elif spec.get('type')=='categorical':
            if any(not isinstance(v,str) for v in observed):_fail('INVALID_FEATURE_TYPE','Categorical observations must be strings')
            missing_policy=spec.get('missing_policy');unknown=spec.get('unknown_policy')
            if missing_policy not in ('error','separate','mode') or unknown not in ('error','separate'): _fail('INVALID_SCHEMA','Declare missing/unknown category policies')
            levels=sorted(set(observed));counts=[observed.count(level) for level in levels]
            if missing_policy=='error' and missing:_fail('MISSING_INPUT',f'{name} missing category')
            if missing_policy=='mode' and not levels:_fail('NO_OBSERVED_VALUES','Cannot estimate categorical mode')
            mode=levels[int(np.argmax(counts))] if levels else None
            reference=spec.get('drop_reference')
            if reference is not None and reference not in levels:_fail('INVALID_SCHEMA','Dropped reference must be a training category')
            fitted.update(levels=levels,counts=counts,missing_policy=missing_policy,unknown_policy=unknown,mode=mode,drop_reference=reference)
            columns=[name+'::level:'+str(i) for i,level in enumerate(levels) if level!=reference]
            if missing_policy=='separate':columns.append(name+'::missing')
            if unknown=='separate':columns.append(name+'::unknown')
        else:_fail('INVALID_SCHEMA','Feature type must be numeric or categorical')
        fitted['output_columns']=columns;specs.append(fitted);output.extend(columns)
        lineage.extend({'output':col,'input':name} for col in columns)
    if len(set(output))!=len(output) or not output:_fail('INVALID_SCHEMA','Output column names collide or design is empty')
    parameters={'feature_specs':specs,'output_order':output,'lineage':lineage}
    model=_artifact('features.fit_preprocessor',names,parameters,target,training,model_id,model_version)
    diagnostics={'n_training':len(rows),'training_input_fingerprint':_fingerprint([{k:None if _missing(row[k]) else row[k] for k in names} for row in rows]),'fit_scope':'TRAINING_ONLY','output_order':output}
    model['diagnostics']=diagnostics
    return _fit_result(model,'features.fit_preprocessor',diagnostics=diagnostics,context=context)


def transform_features(features,fitted_preprocessor,*,context=None):
    """Apply a frozen preprocessor or monotone-bin ModelArtifact without refitting."""
    rows=_records(features);model=_model_artifact(fitted_preprocessor);params=model['parameters'];names=model['feature_order']
    if any(any(name not in row for name in names) for row in rows):_fail('MISSING_FEATURE','Required input column absent')
    matrix=[];missing_counts={name:0 for name in names};unknown_counts={name:0 for name in names}
    if model['method_id']=='features.fit_monotone_bins':
        name=names[0];boundaries=params['upper_boundaries'];missing_index=len(boundaries)
        for row in rows:
            v=row[name]
            if _missing(v):index=missing_index;missing_counts[name]+=1
            else:index=int(np.searchsorted(boundaries[:-1],_number(v,name),side='left'))
            matrix.append([float(index)])
        output=[name+'::bin'];lineage=[{'output':output[0],'input':name}]
    elif model['method_id']=='features.fit_preprocessor':
        output=params['output_order'];lineage=params['lineage']
        for row in rows:
            result=[]
            for spec in params['feature_specs']:
                name=spec['name'];value=row[name];missing=_missing(value)
                if missing:missing_counts[name]+=1
                if spec['type']=='numeric':
                    if missing and spec['imputation']=='error':_fail('MISSING_INPUT',f'{name} missing')
                    v=spec['fill_value'] if missing else _number(value,name)
                    result.append((v-spec['center'])/spec['divisor'])
                    if spec['missing_indicator']:result.append(float(missing))
                else:
                    if missing and spec['missing_policy']=='error':_fail('MISSING_INPUT',f'{name} missing')
                    if not missing and not isinstance(value,str):_fail('INVALID_FEATURE_TYPE','Categorical values must be strings')
                    if missing and spec['missing_policy']=='mode':value=spec['mode'];missing=False
                    unknown=not missing and value not in spec['levels']
                    if unknown:
                        unknown_counts[name]+=1
                        if spec['unknown_policy']=='error':_fail('UNKNOWN_CATEGORY',f'{name} category absent from training')
                    result.extend(float(not missing and value==level) for level in spec['levels'] if level!=spec['drop_reference'])
                    if spec['missing_policy']=='separate':result.append(float(missing))
                    if spec['unknown_policy']=='separate':result.append(float(unknown))
            matrix.append(result)
    else:_fail('UNSUPPORTED_MODEL','Expected feature preprocessing or binning artifact')
    design=np.asarray(matrix,float)
    if np.any(~np.isfinite(design)):_fail('NUMERICAL_RANGE','Feature transformation produced nonfinite values')
    return _result({'design_matrix':design,'feature_order':output,'lineage':lineage},'features.transform_features','design_matrix',context,
        diagnostics={'model_id':model['model_id'],'model_version':model['model_version'],'missing_counts':missing_counts,'unknown_counts':unknown_counts,'refitted':False})


def fit_monotone_bins(x_train,y_train,*,feature_name,direction,min_bin_count,max_prebins,target,training,model_id,model_version,context=None):
    """Train tied-value prebins, merge minimum support then pool monotonic violations."""
    if not isinstance(feature_name,str) or not feature_name:_fail('INVALID_SCHEMA','feature_name required')
    if direction not in ('increasing','decreasing'):_fail('INVALID_INPUT','Explicit monotonic direction required')
    minimum=_number(min_bin_count,'min_bin_count');maximum=_number(max_prebins,'max_prebins')
    if minimum<1 or maximum<1 or minimum!=int(minimum) or maximum!=int(maximum):_fail('INVALID_INPUT','Positive integer support and prebin count required')
    if not isinstance(x_train,(list,tuple,np.ndarray)):_fail('INVALID_INPUT','Feature vector required')
    y=_numeric(y_train,'y_train',vector_only=True)
    if len(x_train)!=len(y) or np.any((y!=0)&(y!=1)):_fail('INVALID_OUTCOME','Equal-length observed binary outcomes required')
    pairs=[];missing_d=missing_n=0
    for x,event in zip(x_train,y):
        if _missing(x):missing_n+=1;missing_d+=int(event)
        else:pairs.append((_number(x,'x_train'),int(event)))
    if len(pairs)<minimum:_fail('INSUFFICIENT_SUPPORT','Nonmissing sample smaller than minimum bin support')
    pairs.sort();unique=[]
    for x,event in pairs:
        if unique and unique[-1][0]==x:unique[-1][1]+=1;unique[-1][2]+=event
        else:unique.append([x,1,event])
    bins=[];current=None;target_count=math.ceil(len(pairs)/maximum)
    for x,n,d in unique:
        if current is None:current=[x,x,n,d]
        else:current[1]=x;current[2]+=n;current[3]+=d
        if current[2]>=target_count:bins.append(current);current=None
    if current:bins.append(current)
    def merge(i):
        left,right=bins[i],bins[i+1];bins[i:i+2]=[[left[0],right[1],left[2]+right[2],left[3]+right[3]]]
    while len(bins)>1 and min(b[2] for b in bins)<minimum:
        i=min(range(len(bins)),key=lambda j:(bins[j][2],j))
        if i==0:merge(0)
        elif i==len(bins)-1:merge(i-1)
        else:
            rate=bins[i][3]/bins[i][2];left=abs(rate-bins[i-1][3]/bins[i-1][2]);right=abs(rate-bins[i+1][3]/bins[i+1][2]);merge(i-1 if left<=right else i)
    sign=1 if direction=='increasing' else -1
    i=0
    while i<len(bins)-1:
        if sign*bins[i][3]/bins[i][2]>sign*bins[i+1][3]/bins[i+1][2]:merge(i);i=max(0,i-1)
        else:i+=1
    table=[{'bin_id':i,'count':b[2],'defaults':b[3],'nondefaults':b[2]-b[3],'default_rate':b[3]/b[2],'observed_min':b[0],'observed_max':b[1],'missing':False} for i,b in enumerate(bins)]
    table.append({'bin_id':len(bins),'count':missing_n,'defaults':missing_d,'nondefaults':missing_n-missing_d,'default_rate':missing_d/missing_n if missing_n else None,'observed_min':None,'observed_max':None,'missing':True})
    params={'upper_boundaries':[b[1] for b in bins[:-1]]+[None],'bins':table,'direction':direction,'interval_convention':'right_closed','missing_bin_id':len(bins),'min_bin_count':int(minimum),'max_prebins':int(maximum)}
    model=_artifact('features.fit_monotone_bins',[feature_name],params,target,training,model_id,model_version)
    model['diagnostics']={'fit_scope':'TRAINING_ONLY','missing_bin_exempt_from_monotonicity_and_support':True,'algorithm':'tied_value_prebin_support_merge_pava','optimal_partition_claim':False}
    return _fit_result(model,'features.fit_monotone_bins',diagnostics=model['diagnostics'],context=context)


def compute_woe_iv(bin_counts,*,smoothing,woe_sign,context=None):
    """Smoothed class-conditional WOE and IV with an explicit sign convention."""
    rows=_records(bin_counts);alpha=_number(smoothing,'smoothing')
    if alpha<0 or woe_sign not in ('nondefault_over_default','default_over_nondefault'):_fail('INVALID_INPUT','Nonnegative smoothing and explicit sign required')
    counts=[];ids=[]
    for row in rows:
        identifier=row.get('bin_id')
        if not isinstance(identifier,(str,int)) or isinstance(identifier,bool) or identifier in ids:_fail('INVALID_BIN','Unique string/integer bin_id required')
        ids.append(identifier);d=_number(row.get('defaults'),'defaults');g=_number(row.get('nondefaults'),'nondefaults')
        if min(d,g)<0 or d!=int(d) or g!=int(g) or max(d,g)>2**53-1:_fail('INVALID_COUNTS','Integer class counts required')
        counts.append([d,g])
    counts=np.asarray(counts);totals=counts.sum(axis=0)
    if np.any(totals>2**53-1) or np.any(~np.isfinite(totals+alpha*len(rows))):_fail('INVALID_COUNTS','Count or smoothing total exceeds supported range')
    if np.any(totals<=0):_fail('INSUFFICIENT_CLASS_SUPPORT','Both classes must be observed before smoothing')
    if alpha==0 and np.any(counts==0):_fail('ZERO_CLASS_CELL','Positive smoothing required for zero class cells')
    shares=(counts+alpha)/(totals+alpha*len(rows));table=[];total=0.
    for i,(d,g) in enumerate(shares):
        canonical=math.log(g)-math.log(d);iv=(g-d)*canonical;total+=iv
        table.append({'bin_id':ids[i],'default_share':float(d),'nondefault_share':float(g),'woe':canonical if woe_sign=='nondefault_over_default' else -canonical,'iv_contribution':iv,'zero_default_count':bool(counts[i,0]==0),'zero_nondefault_count':bool(counts[i,1]==0)})
    return _result({'bins':table,'iv':total},'features.compute_woe_iv','information_value',context,diagnostics={'smoothing':alpha,'woe_sign':woe_sign,'smoothing_scope':'this_bin_partition','iv_sign_invariant':True})


def build_financial_ratios(financial_statements,ratio_spec,*,context=None):
    """Ratios from a single aligned statement per row, with explicit unit/domain rules."""
    statements=_records(financial_statements);specs=_records(ratio_spec);output=[]
    names=[]
    for spec in specs:
        if not isinstance(spec.get('name'),str) or not spec['name'] or spec['name'] in names:_fail('INVALID_SCHEMA','Unique ratio name required')
        names.append(spec['name'])
        if spec.get('denominator_policy') not in ('positive','nonzero') or not isinstance(spec.get('output_unit'),str):_fail('INVALID_SCHEMA','Declare denominator policy and output unit')
        for side in ('numerator','denominator'):
            if not isinstance(spec.get(side),Mapping) or not spec[side]:_fail('INVALID_SCHEMA','Formula requires named field coefficients')
    for row in statements:
        for key in ('entity_id','currency','as_of','period_start','period_end'):
            if not isinstance(row.get(key),str) or not row[key]:_fail('INVALID_STATEMENT',f'{key} required')
        dates=[_feature_date(row[k]) for k in ('as_of','period_start','period_end')]
        if dates[1]>dates[2] or dates[2]!=dates[0]:_fail('INVALID_STATEMENT','Require period_start <= period_end == as_of')
        if not isinstance(row.get('values'),Mapping) or not isinstance(row.get('units'),Mapping):_fail('INVALID_STATEMENT','Values and field units required')
        result={'entity_id':row['entity_id'],'as_of':row['as_of'],'currency':row['currency'],'ratios':{}}
        for spec in specs:
            parts=[];units=[];missing=False
            for side in ('numerator','denominator'):
                amount=0.;unit=None
                for field,coefficient in spec[side].items():
                    if field not in row['values'] or field not in row['units']:_fail('MISSING_FEATURE',f'Statement field {field} absent')
                    field_unit=row['units'][field]
                    if not isinstance(field_unit,str) or not field_unit:_fail('INVALID_UNIT','Explicit field unit required')
                    if field_unit not in ('dimensionless',row['currency']):_fail('INCOMPATIBLE_UNIT','Field currency conflicts with statement currency')
                    if unit is not None and field_unit!=unit:_fail('INCOMPATIBLE_UNIT','Cannot add different units')
                    unit=field_unit;weight=_number(coefficient,'coefficient');value=row['values'][field]
                    if _missing(value):missing=True
                    else:amount+=weight*_number(value,field)
                parts.append(amount);units.append(unit)
            if units[0]!=units[1] or spec['output_unit']!='dimensionless':_fail('INCOMPATIBLE_UNIT','Ratio requires matching input units and dimensionless output')
            numerator,denominator=parts
            status='MISSING_INPUT' if missing else 'ZERO_DENOMINATOR' if denominator==0 else 'NONPOSITIVE_DENOMINATOR' if denominator<0 and spec['denominator_policy']=='positive' else 'OK'
            value=numerator/denominator if status=='OK' else None
            if value is not None and not math.isfinite(value):_fail('NUMERICAL_RANGE','Ratio outside finite range')
            result['ratios'][spec['name']]={'value':value,'status':status,'unit':'dimensionless','numerator':None if missing else numerator,'denominator':None if missing else denominator,'formula':dict(spec)}
        output.append(result)
    return _result(output,'features.build_financial_ratios','ratio_table',context,limitations=('All fields in a statement row must share the declared currency, reporting date and period. No automatic currency or accounting-definition conversion.',))


def _feature_date(v):
    try:
        d=date.fromisoformat(v)
        if d.isoformat()!=v:raise ValueError
        return d
    except (TypeError,ValueError):_fail('INVALID_DATE','ISO date required')


def build_distributed_lags(series,observations,lag_spec,*,frequency,gap_policy='null',context=None):
    """Exact calendar lags and changes, respecting vintage availability at each query."""
    records=_records(series);queries=_records(observations);specs=_records(lag_spec)
    if frequency not in ('days','months','years') or gap_policy not in ('null','error'):_fail('INVALID_INPUT','Declare frequency and null/error gap policy')
    names=[]
    for spec in specs:
        if not isinstance(spec.get('name'),str) or not spec['name'] or spec['name'] in names or not isinstance(spec.get('feature'),str):_fail('INVALID_SCHEMA','Unique output names and input features required')
        names.append(spec['name']);lag=_number(spec.get('lag'),'lag')
        if lag<0 or lag!=int(lag) or spec.get('operation') not in ('level','difference','relative_change'):_fail('INVALID_SCHEMA','Nonnegative integer lag and supported operation required')
    facts=[]
    for row in records:
        for key in ('entity_id','feature'):
            if not isinstance(row.get(key),str) or not row[key]:_fail('INVALID_SERIES',f'{key} required')
        observed,available,known=(_feature_date(row.get(k)) for k in ('observed_at','available_at','known_at'))
        if available<observed or known<available:_fail('INVALID_CHRONOLOGY','Require observed <= available <= known')
        value=None if _missing(row.get('value')) else _number(row['value'],'value')
        facts.append((row['entity_id'],row['feature'],observed,available,known,value))
    def shift(d,lag):
        if frequency=='days':return d-timedelta(days=lag)
        months=lag*(12 if frequency=='years' else 1);index=d.year*12+d.month-1-months;year,month0=divmod(index,12)
        if year<1:_fail('INVALID_DATE','Lag precedes supported calendar')
        return date(year,month0+1,min(d.day,calendar.monthrange(year,month0+1)[1]))
    output=[]
    for query in queries:
        eid=query.get('entity_id');asof=_feature_date(query.get('as_of'));cutoff=_feature_date(query.get('known_at'))
        if not isinstance(eid,str) or not eid:_fail('INVALID_ENTITY','entity_id required')
        def value(feature,at):
            candidates=[f for f in facts if f[0]==eid and f[1]==feature and f[2]==at and f[3]<=cutoff and f[4]<=cutoff]
            if not candidates:return None
            candidates.sort(key=lambda f:(f[4],f[3]))
            if len(candidates)>1 and candidates[-1][3:5]==candidates[-2][3:5]:_fail('AMBIGUOUS_VINTAGE','Duplicate equally available facts')
            return candidates[-1][5]
        values={};statuses={}
        for spec in specs:
            lagged=value(spec['feature'],shift(asof,int(spec['lag'])));current=value(spec['feature'],asof) if spec['operation']!='level' else None
            status='MISSING_LAG' if lagged is None else 'MISSING_CURRENT' if current is None and spec['operation']!='level' else 'ZERO_BASE' if lagged==0 and spec['operation']=='relative_change' else 'OK'
            if status!='OK' and gap_policy=='error':_fail('LAG_UNAVAILABLE',f'{spec["name"]}: {status}')
            result=None if status!='OK' else lagged if spec['operation']=='level' else current-lagged if spec['operation']=='difference' else current/lagged-1
            if result is not None and not math.isfinite(result):_fail('NUMERICAL_RANGE','Lag transformation overflow')
            values[spec['name']]=result;statuses[spec['name']]=status
        output.append({'entity_id':eid,'as_of':query['as_of'],'known_at':query['known_at'],'features':values,'statuses':statuses,'eligible':all(v=='OK' for v in statuses.values())})
    return _result(output,'features.build_distributed_lags','feature_table',context,diagnostics={'frequency':frequency,'gap_policy':gap_policy,'calendar_rule':'exact_date_clamped_month_end','no_row_shift':True})
