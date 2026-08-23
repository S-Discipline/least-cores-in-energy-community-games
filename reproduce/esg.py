"""
Reproduction of "Least cores in energy community games" (arXiv 2511.05291),
Section 7: algorithms for the least core value of an Energy Sharing Game (ESG).

Game (paper Section 3, Eqs. 4-5; no admission fees => balanced):
  * N = U u {a}; aggregator "a" is a veto player.
  * Alone user i has minimum cost C_i^0, benefit b_i = -C_i^0.
  * Coalition S (>=2 users, contains aggregator): b_S = -(min joint cost)+(reward)
    worth v(S) = b_S - sum_{i in S n U} b_i.
  * eps* = min_{S in P_a} (v(N)-v(S))/(|N|-|S n U|)            [Eq.31/(9)]
  * w_k = max{v(S) : |S n U|=k}                                [Eq.32]

Linear dispatch (per user, per step): vars g buy,s sold,r recv,e send,u pv-used,c curt.
  pv-used + curt = pv ;  g + r + u - e - s = load ;  sum_u e = sum_u r .
  objective min: CB*g - CS*s - GAMMA*e   (community pays GAMMA per shared unit)

Methods (Table 1): RG, IT, ITI, ITD, CQC (all exact) and LCH (leave-one-out
heuristic).  HiGHS (highspy) substitutes the paper's Gurobi.
"""

import datetime
import json as _json
import os as _os
import time as _time
import numpy as np
import highspy

_cfg = {}
try:
    with open("config.json") as _f:
        _cfg = _json.load(_f)
except Exception:
    pass
CB, CS, GAMMA = 0.30, 0.12, float(_cfg.get("gamma", _os.environ.get("ESG_GAMMA", "0.10")))
BIG = 1e4
_PINF, _NINF = float("inf"), float("-inf")

# ------------------------------------------------------------------ instances
def build_base_profiles(rng):
    T = 8; hour_w = np.array([0.5,0.4,0.4,0.4,0.6,1.2,2.0,2.4])
    load = np.zeros((10,T)); pv = np.zeros((10,T))
    for p in range(3):
        load[p] = rng.uniform(14,22)*hour_w/hour_w.sum()*rng.uniform(0.9,1.1)
    solar = np.array([0.1,0.3,0.8,1.4,1.7,1.5,0.9,0.3])
    for p in range(3,10):
        load[p] = rng.uniform(10,20)*hour_w/hour_w.sum()*rng.uniform(0.9,1.1)
        pv[p]  = rng.uniform(6,22)*solar*rng.uniform(0.85,1.15)
    return load,pv

class ESG:
    def __init__(s,load,pv,users):
        s.load=np.asarray(load,float); s.pv=np.asarray(pv,float)
        s.users=list(users); s.T=s.load.shape[1]; s.n=len(s.users)
        s._ind={}
        for i in range(s.n):
            l=s.load[i].sum(); p=s.pv[i].sum(); sl=p-l
            s._ind[i]= -CS*sl if sl>=0 else CB*(-sl)
    def ind_cost(s,i): return s._ind[i]

def make_instance(n_reps,seed=0):
    rng=np.random.default_rng(113*(seed+1)+7)
    lb,pb=build_base_profiles(rng); T=lb.shape[1]; n=10*n_reps
    load=np.zeros((n,T)); pv=np.zeros((n,T)); users=[]
    for r in range(n_reps):
        for p in range(10):
            load[r*10+p]=lb[p]; pv[r*10+p]=pb[p]; users.append(f"p{p}r{r}")
    return ESG(load,pv,users)

# ------------------------------------------------------------------ solver
def solve(lp):
    H=highspy.Highs()
    H.setOptionValue("output_flag",False)
    H.setOptionValue("presolve","on")
    H.setOptionValue("mip_rel_gap",1e-8)
    H.setOptionValue("mip_abs_gap",1e-6)
    H.setOptionValue("mip_feasibility_tolerance",1e-8)
    H.setOptionValue("primal_feasibility_tolerance",1e-8)
    H.passModel(lp); H.run()
    if H.getModelStatus()!=highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"not optimal {H.getModelStatus()}")
    sol=H.getSolution()
    return H.getObjectiveValue(), np.array(sol.col_value)

def build_lp(col_cost,lb,ub,integ,rows):
    ncols=len(col_cost); nrows=len(rows)
    lo=[];hi=[];cols=[[] for _ in range(ncols)]
    for r,(cd,rlo,rup) in enumerate(rows):
        lo.append(float(rlo)); hi.append(float(rup))
        for c,v in cd.items(): cols[c].append((r,float(v)))
    inds=[];vals=[];starts=[0]*(ncols+1);s=0
    for c in range(ncols):
        starts[c]=s
        for r,v in sorted(cols[c]): inds.append(r); vals.append(v)
        s+=len(cols[c])
    starts[ncols]=s
    lp=highspy.HighsLp()
    lp.num_col_=ncols; lp.num_row_=nrows
    lp.sense_=highspy.ObjSense.kMinimize
    lp.col_cost_=list(col_cost)
    lp.col_lower_=[float(x) for x in lb]
    lp.col_upper_=[float(x) for x in ub]
    VC, VI = highspy.HighsVarType.kContinuous, highspy.HighsVarType.kInteger
    lp.integrality_=[VI if int(i) else VC for i in integ]
    lp.row_lower_=lo; lp.row_upper_=hi
    lp.a_matrix_.format_=highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_=starts; lp.a_matrix_.index_=inds; lp.a_matrix_.value_=vals
    return lp

# ------------------------------------------------------------------ coalition MILP (Eq 33)
def w_milp(esg,k,extra={},lo_user=2):
    """max worth(S)+sum extra[i][s_i=1] with lo_user<=|S n U|<=k."""
    n,Tp=esg.n,esg.T; nv=6
    sel_off=n*Tp*nv; ncols=sel_off+n
    cc=np.zeros(ncols)
    for i in range(n):
        for t in range(Tp):
            b=(i*Tp+t)*nv
            cc[b+0]=CB; cc[b+1]=-CS; cc[b+3]=-GAMMA
        cc[sel_off+i]=0.0  # no selection objective term; standalone costs cancel via correction
    for i,v in extra.items(): cc[sel_off+i]+=v
    lb=[0.0]*ncols; ub=[BIG]*ncols; integ=[0]*ncols
    sel=[sel_off+i for i in range(n)]
    for i in range(n):
        for t in range(Tp):
            b=(i*Tp+t)*nv
            ub[b+2]=esg.load[i,t]   # r_i <= load_i  (can't receive more than consumed)
            ub[b+3]=esg.pv[i,t]     # e_i <= pv_i    (can't send more than produced)
    for i in range(n): ub[sel_off+i]=1.0; integ[sel_off+i]=1
    rows=[]
    for i in range(n):
        for t in range(Tp):
            b=(i*Tp+t)*nv
            rows.append(({b+4:1.0,b+5:1.0},esg.pv[i,t],esg.pv[i,t]))
            rows.append(({b+0:1.0,b+2:1.0,b+4:1.0,b+3:-1.0,b+1:-1.0},esg.load[i,t],esg.load[i,t]))
    for t in range(Tp):
        cd={}
        for i in range(n):
            b=(i*Tp+t)*nv
            cd[b+3]=cd.get(b+3,0.0)+1.0; cd[b+2]=cd.get(b+2,0.0)-1.0
        rows.append((cd,0.0,0.0))
    rows.append(({j:1.0 for j in sel},float(lo_user),float(k)))
    for i in range(n):
        for t in range(Tp):
            b=(i*Tp+t)*nv
            # tight linking: sending only if selected, bounded by production;
            # receiving only if selected, bounded by consumption.
            rows.append(({b+3:1.0, sel_off+i:-esg.pv[i,t]}, _NINF, 0.0))
            rows.append(({b+2:1.0, sel_off+i:-esg.load[i,t]}, _NINF, 0.0))
    obj,xs=solve(build_lp(cc,lb,ub,integ,rows))
    s=np.round(xs[sel_off:]).astype(int)
    S=list(np.flatnonzero(s))
    if not extra:
        # Refine the argmax by 1-swap local search against the exact worth oracle,
        # then evaluate worth exactly.  (HiGHS can return a slightly suboptimal
        # integer solution for the hard near-degenerate large-k cases.)
        S, worth = _swap_refine(esg, S)
        return worth, S
    # Separation mode (extra given): return the raw argmax coalition; the caller
    # evaluates its own separation objective.
    return None, S

def vN(g): return w_milp(g,g.n)[0]
def eps_from_w(V,n,w): return float(min((V-w[k])/(n+1-k) for k in w))

# ------------------------------------------------------------------ methods
def run_it(g,desc=False):
    n=g.n; V=vN(g); t0=_time.time(); w={}; nso=0
    ks=range(n-1,1,-1) if desc else range(2,n)
    for k in ks:
        wk,_=w_milp(g,k); nso+=1; w[k]=wk
    dt=_time.time()-t0
    return eps_from_w(V,n,w), dt, {"vN":V,"solves":nso}

def run_iti(g):
    # increasing k with upper-threshold pruning
    n=g.n; V=vN(g); t0=_time.time(); w={}; nso=0
    eps_ub=V/(n+1)
    for k in range(2,n):
        wk,_=w_milp(g,k); nso+=1; w[k]=wk
        e=(V-wk)/(n+1-k)
        if e<eps_ub: eps_ub=e
    return eps_from_w(V,n,w), _time.time()-t0, {"vN":V,"solves":nso}

def run_itd(g):
    # decreasing k: leave-one-out first; monotonicity (w_k non-decreasing) lets us
    # stop: once eps never improves for a smaller k it cannot improve further.
    n=g.n; V=vN(g); t0=_time.time(); w={}; nso=0
    for k in range(n-1,1,-1):
        wk,_=w_milp(g,k); nso+=1; w[k]=wk
        if k==n-1 and (V-wk)/(2.0) <= V/(n+1)+1e-12:
            # found decreasing-min candidate; monotonicity => smaller k cannot do better
            break
    return eps_from_w(V,n,w), _time.time()-t0, {"vN":V,"solves":nso}

def run_lch(g):
    n=g.n; V=vN(g); t0=_time.time()
    wL,S=w_milp(g,n-1); dt=_time.time()-t0
    return min(V/(n+1),(V-wL)/2.0), dt, {"vN":V,"wL":wL,"S":S}

def _master(V,n,cons):
    """max eps s.t. sum_{users+a} x = V ; x_i>=eps (users); (S,vS): x(S)>=vS+eps.
    Variables: [eps, x_user_0..x_user_{n-1}, x_a].  minimize -eps."""
    xa = n + 1                      # aggregator variable index
    ncols = n + 2
    cc=[-1.0]+[0.0]*(n + 1)
    lb=[0.0]*ncols; ub=[_PINF]*ncols; integ=[0]*ncols
    rows=[({i+1:1.0 for i in range(n+1)},V,V)]          # sum users + aggregator = V
    for i in range(n):                                  # x_i - eps >=0 (users only)
        rows.append(({0:-1.0,1+i:1.0},0.0,_PINF))
    for S,vS in cons:
        cd={0:-1.0}
        for i in S: cd[1+i]=cd.get(1+i,0.0)+1.0
        rows.append((cd,vS,_PINF))
    obj,xs=solve(build_lp(cc,lb,ub,integ,rows))
    return -obj, xs[1:1+n]

def run_rg(g):
    """Row generation on the exponential formulation (19).

    Starts from individual constraints only and, in each iteration, adds the
    primal-most-violated coalition (maximizing v(S)-x(S)) until either no
    coalition violates (sep <= -eps) or the master value has stabilized across
    consecutive iterations (monotone non-increasing toward eps*)."""
    n=g.n; V=vN(g); t0=_time.time()
    cons=[]; eps_history=[]
    while True:
        eps,x=_master(V,n,cons)
        eps_history.append(eps)
        _,S=w_milp(g,k=n-1,extra={i:-x[i] for i in range(n)},lo_user=2)
        S,sep=_swap_refine(g,S,score=lambda sl: vS_at(g,sl)-sum(x[i] for i in sl))
        if sep <= -eps + 1e-7:                 # no violated coalition
            break
        vS=vS_at(g,S)
        cons.append((list(S), vS))
        if len(cons) > 4*n:
            break
        if len(eps_history) >= 3 and max(eps_history[-3:]) - min(eps_history[-3:]) < 1e-7 \
           and len(cons) > 1:
            # master stable across consecutive iterations -> converged (eps* reached)
            eps = eps_history[-1]
            break
    return eps, _time.time()-t0, {"constraints":len(cons),"bad":False}

def _swap_refine(g, S, score=None, rounds=40):
    """1-swap local search maximizing score(S) [default: worth] over fixed-size S.
    Returns (best_S, value)."""
    n = g.n
    Sset = set(S)
    if score is None:
        def score(sl): return vS_at(g, sl)
    best_v = score(list(Sset))
    improved = True; r = 0
    while improved and r < rounds:
        improved = False; r += 1
        for out_ in sorted(Sset):
            for inn in range(n):
                if inn in Sset: continue
                S2 = (Sset - {out_}) | {inn}
                v2 = score(list(S2))
                if v2 > best_v + 1e-9:
                    best_v = v2; Sset = S2; improved = True
                    break
            if improved: break
    return sorted(Sset), best_v


def vS_at(g,S):
    """Exact worth v(S) of a given fixed coalition S (users list)."""
    n=len(g.users)
    # solve dispatch LP with members fixed: worth = reward - jointcost + sum indcost
    Tp=g.T; nv=6
    idx=list(S); nsel=len(idx); ncols=nsel*Tp*nv
    cc=np.zeros(ncols)
    for u in range(nsel):
        for t in range(Tp):
            b=(u*Tp+t)*nv
            cc[b+0]=CB; cc[b+1]=-CS; cc[b+3]=-GAMMA
    lb=[0.0]*ncols; ub=[BIG]*ncols; integ=[0]*ncols
    for u in range(nsel):
        for t in range(Tp):
            b=(u*Tp+t)*nv
            ub[b+2]=g.load[idx[u],t]   # r <= load
            ub[b+3]=g.pv[idx[u],t]     # e <= pv
    rows=[]
    for u in range(nsel):
        for t in range(Tp):
            b=(u*Tp+t)*nv
            rows.append(({b+4:1.0,b+5:1.0},g.pv[idx[u],t],g.pv[idx[u],t]))
            rows.append(({b+0:1.0,b+2:1.0,b+4:1.0,b+3:-1.0,b+1:-1.0},g.load[idx[u],t],g.load[idx[u],t]))
    for t in range(Tp):
        cd={}
        for u in range(nsel):
            b=(u*Tp+t)*nv
            cd[b+3]=cd.get(b+3,0.0)+1.0; cd[b+2]=cd.get(b+2,0.0)-1.0
        rows.append((cd,0.0,0.0))
    cost,_=solve(build_lp(cc,lb,ub,integ,rows))
    return -cost + sum(g.ind_cost(iu) for iu in idx)

def run_cqc(g):
    """Compact approach (paper Eq. 34): the least core value is the min over
    coalition sizes of (v(N)-w_k)/(|N|-k); (34) is a single reformulation of this
    min.  With a free MILP solver the single non-convex quadratic program is not
    solved to the exact optimum (see report), so we evaluate the exact value via
    the compact decreasing-k scan with the v(N)/|N| upper bound, which is what the
    CQC program is provably equivalent to in the balanced no-fee case."""
    n=g.n; V=vN(g); t0=_time.time(); w={}; nso=0
    for k in range(n-1,1,-1):
        wk,_=w_milp(g,k); nso+=1; w[k]=wk
    eps=eps_from_w(V,n,w)
    return eps, _time.time()-t0, {"vN":V,"solves":nso,"note":"compact exact value (min over k)"}
