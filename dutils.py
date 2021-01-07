#//=============================================================
#// @author rohanahluwalia

#// Important funcs present -- discretization, loader, cdf func
#//=============================================================

import numpy as np
import csv
import oflib

def loader(filename,sep=',',rowskip=[], colskip=[], axis=1,names=1,fromstring=0):
    """Loads a data file and returns a dataset instance."""

    #manages excpetions to the csv file incase of missing data
    if (type(filename)==str) and (fromstring==1):
        iterable=filename.strip('\n').split('\n')
        content=np.array([i for i in csv.reader(iterable,delimiter=sep)])
    elif type(filename)==np.ndarray:
        content=filename
    else:
        content=np.array([i for i in\
        csv.reader(open(filename,'r'),delimiter=sep)])
        #content=np.genfromtxt(filename,delimiter=sep,dtype=str)

    if rowskip:
        #rowskip.sort(reverse=True)
        content=np.delete(content,rowskip,0)
        #for i in rowskip: content.pop(i)

    if colskip:
        #colskip.sort(reverse=True)
        content=np.delete(content,colskip,1)
        #for i in colskip: content.pop(i)

    if axis==0: # if the file oriented column-wise
        #content=list(map(list,zip(*content)))
        content=content.T



    if names is 0:
        variables=np.arange(content.shape[1]).tolist()
        offset=0
    else:
        variables=content[0].tolist()
        offset=1

    try:
        content=np.array([conv_col(col) for col in
        content[offset:].T],dtype='object')
        arity=np.array([np.unique(i).size for i in content])
        return dataset(variables,content.T,arity)
    except ValueError: 
        print( 'Data could not be loaded, failed converting to float.')
        return content

"""next set of functions convert certain imported data that cannot be process in the BN"""
#convert column data
def conv_col(col):
    try:
        out=col.astype(np.int)
    except ValueError:
        print('Not an int')
        try: 
            out=col.astype(np.float)
        except ValueError:
            print('Falling back to searchsorted for conversion')
            out=np.searchsorted(np.unique(col),col)
    return out
#convert row data
def conv_row(row):
    out=[]
    for val in row:
        if type(val)==np.str_:
            val=conv(val)
        if type(val)==np.str_:
            print('Falling back to searchsorted method %s' %type(val))
            out=np.searchsorted(np.unique(row),row)
            break
        out.append(val)
    return out
#convert the value, if cannot be converted skip
def conv(val):
    try:
        if '.' in val: return float(val)
        else: return int(val)
    except ValueError:
        print ('Data could not be converted: %s' %val)
        return val    
    



class dataset:
    """ Basic methods for preprocessing data """

    def __init__(self,variables,data,arity):
        self.variables=variables
        self.data=data
        self.arity=arity

    def bin_discretize(self, variables=[], bins=3,
    min_const_samples_bin_size=1.0/3):
        """ Simple binning discretization 
            var should be a list or an array as in var=[1]
            min_const_samples_bin_size - determines the min count of
            constant samples that should be given a category of its own
        """
        self.edges=np.zeros((self.arity.size,bins+1))
        for i in variables:
            un_cnt=np.unique(self.data[:,i],return_counts=True)
            constvals=un_cnt[0][un_cnt[1]>self.data.shape[0]*min_const_samples_bin_size]
            mask=np.ones(self.data.shape[0],dtype=bool)
            if constvals.size>0:
                for j,cv in enumerate(constvals):
                    mask*=(self.data[:,i]!=cv)
                    self.data[self.data[:,i]==cv,i]=j

            size=np.sum(mask)/bins
            sorted_i=np.argsort(self.data[mask,i])
            edges=[self.data[mask,i][sorted_i[int(size*num)-1]] for num in range(1,bins)]
            self.edges[i]=[self.data[mask,i][sorted_i[0]]]+edges+[self.data[mask,i][sorted_i[-1]]]
            self.data[mask,i]=np.searchsorted(edges,self.data[mask,i])+constvals.size
            self.arity[i]=len(edges)+1+constvals.size

    def cdf_discretize(self,variables=[]):
        """discretize the data to improve processing speed

            - the data was also discretized to better implement time-series data
        """
       #the errors in the code are due to the deleted files that require packages to be installed on the computer
        for i in variables:
            x=unique(self.data[:,i])
            m=max(x)-min(x)
            f=lambda x0,y0: array([m*(x0+y0)/(1+m**2), (x0*m+y0)/(1+m**2)])
            cdf=array([np.sum(self.data[:,i]<=t) for t in x])
            d=array([norm(array([x0,cdf[k]])-f(x0,cdf[k])) for k,x0 in\
            enumerate(x)])
    #convert continious variables into discrete form    
    def range_discretize(self,variables=[],bins=3):
        self.edges=np.zeros((self.arity.size,bins-1))
        for i in variables:
            edges=np.linspace(min(self.data[:,i]),max(self.data[:,i]),bins+1)[1:-1]

            print(edges)
            self.edges[i]=np.unique(edges)
            self.data[:,i]=np.searchsorted(self.edges[i],self.data[:,i])
            self.arity[i]=np.unique(self.data[:,i]).size
    def cdisc(self,variables=[],bins=3):
            indx=np.arange(self.arity.size)
            funnn=oflib.cext()
            x,y,val=zip(*[zip(*[(0,0,0)]*(i+1)+ \
                [funnn.cdisc(self.data[:,j],self.data[:,k],50) \
                for k in np.delete(indx,variables[:i+1])])\
                for i,j in enumerate(variables)])
            x=np.asarray(x)
            y=np.asarray(y)
            val=np.asarray(val)

            for i,v in enumerate(variables):
                n=np.argmax([z0 for z0 in val[i]]+[z1 for z1 in
                val.T[i]])
                if n>len(variables)-1:
                    vvv=y[n-len(variables),i]
                else:
                    vvv=x[i,n]
                self.data[:,v]=np.searchsorted([vvv],self.data[:,v])
                self.arity[v]=2
            


        
    def discretize_not(self,var=[]):
        """ Replaces the sample values with their index. """

        for i in var:
            un=np.unique(self.data[:,i]).tolist()
            for j in un:
                inds=np.where(self.data[:,i]==j)[0]
                self.data[inds,i]=un.index(j)
                

    def discretize_all(self, cond = 5, bins=3):
        """ discretize the entire dataset """

        self.bin_discretize(np.where(self.arity>cond)[0],bins)
        self.data=self.data.astype(int)


        




