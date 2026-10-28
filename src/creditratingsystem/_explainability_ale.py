"""Finite-bin first and purified second-order accumulated local effects."""
import numpy as np
from .core import _fail,_numeric,_result,_fingerprint
from .portfolio import _ids,_weights,_sum
from .explainability import _ex_contract,_ex_predict


def compute_credit_ale(predictor,X,model_contract,*,selected_features,bins,feature_types,weights=None,min_bin_count=1,context=None):
    """Accumulate weighted local differences only across fully supported numeric bins."""
    features,ctx=_ex_contract(model_contract,context);selected=_ids(selected_features,'selected_features');x=_numeric(X,'X')
    if len(selected)not in (1,2)or set(selected)-set(features)or x.ndim!=2 or x.shape[1]!=len(features):_fail('ALE_SHAPE','One or two selected features and aligned reference matrix required')
    if not isinstance(feature_types,dict)or any(feature_types.get(f)!='continuous'for f in selected):_fail('ALE_FEATURE_TYPE','Selected numeric features require explicit continuous type; no categorical interpolation')
    if not isinstance(bins,dict)or set(bins)!=set(selected)or type(min_bin_count)is not int or min_bin_count<1:_fail('ALE_BINS','Explicit bins for selected features and positive support count required')
    w=_weights([1]*len(x)if weights is None else weights,len(x));edges=[];indices=[];coordinates=[];positions=[]
    for f in selected:
        j=features.index(f);b=_numeric(bins[f],'bins',vector_only=True)
        if len(b)<2 or np.any(np.diff(b)<=0)or not np.all(np.isfinite(np.diff(b)))or np.any(x[:,j]<b[0])or np.any(x[:,j]>b[-1]):_fail('ALE_BINS','Increasing finite edges must cover all reference values')
        k=np.minimum(np.searchsorted(b,x[:,j],side='right')-1,len(b)-2);t=(x[:,j]-b[k])/(b[k+1]-b[k]);edges.append(b);indices.append(k);coordinates.append(t);positions.append(j)
    shape=tuple(len(b)-1 for b in edges);effects=np.full(shape,np.nan);counts=np.zeros(shape,dtype=int);masses=np.zeros(shape);unsupported=[]
    for cell in np.ndindex(shape):
        member=np.ones(len(x),dtype=bool)
        for d,k in enumerate(cell):member&=indices[d]==k
        usable=member&(w>0);counts[cell]=int(usable.sum());masses[cell]=_sum(w[usable])
        if counts[cell]<min_bin_count or masses[cell]<=0:unsupported.append(list(cell));continue
        z=x[usable];cw=w[usable]/masses[cell]
        if len(selected)==1:
            low=z.copy();high=z.copy();low[:,positions[0]]=edges[0][cell[0]];high[:,positions[0]]=edges[0][cell[0]+1];delta=_ex_predict(predictor,high,model_contract)-_ex_predict(predictor,low,model_contract)
        else:
            results={}
            for a,b in ((0,0),(1,0),(0,1),(1,1)):
                corner=z.copy();corner[:,positions[0]]=edges[0][cell[0]+a];corner[:,positions[1]]=edges[1][cell[1]+b];results[a,b]=_ex_predict(predictor,corner,model_contract)
            delta=results[1,1]-results[1,0]-results[0,1]+results[0,0]
        effects[cell]=_sum(cw*delta)
    curve=None;reference=None;center=None;purification=None
    if not unsupported:
        if len(selected)==1:
            accumulated=np.r_[0,np.cumsum(effects)];raw=accumulated[indices[0]]+coordinates[0]*effects[indices[0]];center=_sum(w*raw);curve=(accumulated-center).tolist();reference=(raw-center).tolist()
        else:
            accumulated=np.zeros((shape[0]+1,shape[1]+1));accumulated[1:,1:]=np.cumsum(np.cumsum(effects,axis=0),axis=1)
            def surface(ki,kj,ti,tj):return ((1-ti)*(1-tj)*accumulated[ki,kj]+ti*(1-tj)*accumulated[ki+1,kj]+(1-ti)*tj*accumulated[ki,kj+1]+ti*tj*accumulated[ki+1,kj+1])
            ki,kj=indices;ti,tj=coordinates;dx=np.zeros(shape[0]);dy=np.zeros(shape[1])
            for k in range(shape[0]):
                mask=(ki==k)&(w>0);cw=w[mask]/_sum(w[mask]);dx[k]=_sum(cw*(surface(ki[mask],kj[mask],np.ones(mask.sum()),tj[mask])-surface(ki[mask],kj[mask],np.zeros(mask.sum()),tj[mask])))
            for k in range(shape[1]):
                mask=(kj==k)&(w>0);cw=w[mask]/_sum(w[mask]);dy[k]=_sum(cw*(surface(ki[mask],kj[mask],ti[mask],np.ones(mask.sum()))-surface(ki[mask],kj[mask],ti[mask],np.zeros(mask.sum()))))
            gx=np.r_[0,np.cumsum(dx)];gy=np.r_[0,np.cumsum(dy)];raw=surface(ki,kj,ti,tj)-(gx[ki]+ti*dx[ki])-(gy[kj]+tj*dy[kj]);center=_sum(w*raw)
            curve=(accumulated-gx[:,None]-gy[None,:]-center).tolist();reference=(raw-center).tolist();purification=dict(first_axis_accumulated=gx.tolist(),second_axis_accumulated=gy.tolist())
    local=effects.astype(object)
    for cell in np.ndindex(shape):
        if not np.isfinite(effects[cell]):local[cell]=None
    selected_values=x[:,positions];means=w@selected_values;centered=selected_values-means;cov=(centered*w[:,None]).T@centered;sd=np.sqrt(np.diag(cov));corr=[[None if sd[i]==0 or sd[j]==0 else float(cov[i,j]/(sd[i]*sd[j]))for j in range(len(selected))]for i in range(len(selected))]
    return _result(dict(status='NON_ESTIMABLE_SUPPORT'if unsupported else'ESTIMABLE',order=len(selected),selected_features=selected,bin_edges={f:edges[i].tolist()for i,f in enumerate(selected)},positive_weight_counts=counts.tolist(),bin_masses=masses.tolist(),local_differences=local.tolist(),unsupported_cells=unsupported,centered_ale_grid=curve,reference_ale=reference,centering_constant=center,purification=purification,weighted_selected_correlation=corr,min_bin_count=min_bin_count,interpolation='WITHIN_BIN_LINEAR_OR_BILINEAR_NO_GAP_FILLING'),'explainability.compute_credit_ale',model_contract['output_scale'],ctx,diagnostics=dict(model_contract_hash=_fingerprint(model_contract),reference_hash=_fingerprint(dict(X=x.tolist(),weights=w.tolist())),bins_hash=_fingerprint(bins)),limitations=('Local finite differences use frozen predictions at declared bin boundaries. First-order effects are accumulated with within-bin linear interpolation and weighted empirical centering. Second-order effects use mixed differences, double accumulation and removal of both first-order accumulated components before centering. Empty or insufficient positive-weight cells make the global curve non-estimable; supported local differences remain available. Correlation and counts do not establish off-manifold boundary support or causal effects. Output effects use the declared prediction scale.',))
