#//=============================================================
#// @author rohanahluwalia

#// Different optimization method for bayes networks
#// These optimization methods were from literature but are just implemented here
#//=============================================================

import numpy as np
from itertools import product
	

def mdl(data,arity):
	""" Minimum Description Length metric as described by Decampos"""

	ri=arity[0]
	cnode=data[:,0]

	if data.shape[1]==1:
		abasis=np.array([0])
	else:
		arity[0]=1
		abasis=np.concatenate(([0],np.multiply.accumulate(arity[:-1])))
		arity[0]=ri

	drepr=np.dot(data,abasis)
	un=np.unique(drepr)
	w_ij=dict([(val,ind) for ind,val in enumerate(un)])
	v_ik=np.unique(cnode)
	Nijk=np.zeros((un.size,ri),dtype=int)

	for j,k in product(un,v_ik):
		Nijk[w_ij[j],k]=np.dot((drepr==j).astype(int),(cnode==k).astype(int))

	N_ij=np.sum(Nijk,axis=1)
	N_ij=N_ij[N_ij.nonzero()]
	Nijk=Nijk[Nijk.nonzero()]
	H=np.dot(Nijk,np.log(Nijk))-np.dot(N_ij,np.log(N_ij))
	complexity=(ri-1)*np.multiply.reduce(arity[1:])
	return H-complexity*np.log(cnode.size)*.5


def cpt(data,arity):
	""" Conditional Probability table"""

	ri=arity[0]
	cnode=data[:,0]

	#anaylze data shape
	if data.shape[1]==1:
		abasis=np.array([0])
	else:
		arity[0]=1
		abasis=np.concatenate(([0],np.multiply.accumulate(arity[:-1])))
		arity[0]=ri

	drepr=np.dot(data,abasis)
	un=np.unique(drepr)
	w_ij=dict([(val,ind) for ind,val in enumerate(un)])
	v_ik=np.unique(cnode)
	Nijk=np.zeros((un.size,ri),dtype=int)

	for j,k in product(un,v_ik):
		Nijk[w_ij[j],k]=np.dot((drepr==j).astype(int),(cnode==k).astype(int))
	N_ij=np.sum(Nijk,axis=1)
	#N_ij=N_ij[N_ij.nonzero()]
	N_ijk=Nijk[Nijk.nonzero()]
	H=np.dot(N_ijk,np.log(N_ijk))-np.dot(N_ij,np.log(N_ij))
	complexity=(ri-1)*np.multiply.reduce(arity[1:])

	pstates=[data[np.where(drepr==q)[0][0],1:] for q in un]

	return Nijk,N_ij,pstates, H,complexity

def bdm(data,arity):
	""" Bayesian Dirichlet metric as described by Cooper&Herskovits """

	csize=data.shape[0]+max(arity)
	lgcache=np.arange(0,csize)
	lgcache[0]=1
	lgcache=np.log(lgcache)
	lgcache[0]=0.0
	lgcache=np.add.accumulate(lgcache)
	ri=arity[0]
	cnode=data[:,0]

	if data.shape[1]==1:
		abasis=np.array([0])
	else:
		arity[0]=1
		abasis=np.concatenate(([0],np.multiply.accumulate(arity[:-1])))
		arity[0]=ri

	drepr=np.dot(data,abasis)
	un=np.unique(drepr)
	w_ij=dict([(val,ind) for ind,val in enumerate(un)])
	v_ik=np.unique(cnode)
	Nijk=np.zeros((un.size,ri),dtype=int)
	for j,k in product(un,v_ik):
		Nijk[w_ij[j],k]=np.dot((drepr==j).astype(int),(cnode==k).astype(int))

	N_ij=np.sum(Nijk,axis=1)
	return np.sum(lgcache[ri-1]+np.sum(lgcache[Nijk],axis=1)-lgcache[N_ij+ri-1])




try:
	from scipy.weave import inline

	def mdl_c(data,arity):
		""" MDL with C inlining using scipy.weave """

		ri=arity[0]
		child=data[:,0]

		if data.shape[1]==1:
			abasis=np.array([0])
		else:
			arity[0]=1
			abasis=np.concatenate(([0],np.multiply.accumulate(arity[:-1])))
			arity[0]=ri

		drepr=np.dot(data,abasis)
		u=np.unique(drepr)
		drepr=np.searchsorted(u,drepr)
		Nijk=np.zeros((u.size,ri),dtype=int)
		nsamples=child.size
		ri=int(ri)

		code="""
			for (int i=0; i<nsamples; ++i){
				Nijk[drepr[i]*ri+child[i]]+=1;
				}
			"""
		inline(code,['Nijk','drepr','child','nsamples','ri'])

		II=np.ones(ri)
		N_ij=np.dot(Nijk,II)
		N_ij=N_ij[N_ij.nonzero()]
		Nijk=Nijk[Nijk.nonzero()]
		H=np.dot(Nijk,np.log(Nijk))-np.dot(N_ij,np.log(N_ij))
		return H-(ri-1)*np.multiply.reduce(arity[1:])*np.log(nsamples)*0.5

	def bdm_c(data,arity):
		""" Same as bdm except with scipy.weave C inlining """

		csize=data.shape[0]+max(arity)
		lgcache=np.arange(0,csize)
		lgcache[0]=1
		lgcache=np.log(lgcache)
		lgcache[0]=0.0
		lgcache=np.add.accumulate(lgcache)
		ri=arity[0]
		child=data[:,0]

		if data.shape[1]==1:
			abasis=np.array([0])
		else:
			arity[0]=1
			abasis=np.concatenate(([0],np.multiply.accumulate(arity[:-1])))
			arity[0]=ri

		drepr=np.dot(data,abasis)
		u=np.unique(drepr)
		drepr=np.searchsorted(u,drepr)
		Nijk=np.zeros((u.size,ri),dtype=int)
		nsamples=child.size
		ri=int(ri)

		code="""
			for (int i=0; i<nsamples; ++i){
				Nijk[drepr[i]*ri+child[i]]+=1;
				}
			"""
		inline(code,['Nijk','drepr','child','nsamples','ri'])

		N_ij=np.sum(Nijk,axis=1)
		return np.sum(lgcache[ri-1]-lgcache[N_ij+ri-1]+np.sum(lgcache[Nijk],axis=1))

except ImportError: pass
