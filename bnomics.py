#//=============================================================
#// @author rohanahluwalia

#// defines p_score --- relation to edge creation in markov blanket
#// search class - main section fo bn learning
#//=============================================================




import os
import textwrap as tw
import dutils
from ofunc import bdm, mdl, cpt
from bnutils import bnet, np
from copy import deepcopy

#This section runs the make file, which I commented out to remove the fatal errors in the algorithm without the proper packages
# for solution output as pdf(ex. graphviz)
"""
try: from ofunc import bdm_c, mdl_c
except ImportError: pass
try: 
    import oflib
    cmdla=oflib.cext().cmdla
    cmdlb=oflib.cext().cmdlb
except OSError: 
    print("Try running make")
    pass
"""


class search:
        """ BN structure learning methods"""

        def __init__(self, dt, objfunc='bdm',cache_size=1):
                """
                dt: dataset instance of dutils.dataset
                objfunc: 'bdm' or 'mdl', defalut is 'bdm' 
                If C extension is properly compiled 'cmdla' (AIC) and
                'cmdlb' (BIC)options may also be availabe
                                
                """

                self.objfunc=eval(objfunc)

                self.data=dt.data
                self.arity=dt.arity
                self.variables=dt.variables

                self.BN=bnet(self.variables)

                node_index=np.asarray([i for i,j in enumerate(self.variables)])
                self.node_index=node_index

                self.scores=[self.objfunc(self.data[:,[i]],self.arity[[i]]) for i in node_index]
                self.net_score=np.sum(self.scores)

                self.cache_size=self.arity.size
                if cache_size:
                        self.cache_size=cache_size


                self.delta_cache=np.zeros((node_index.size,self.cache_size))
                self.delta_index=np.zeros((node_index.size,self.cache_size),dtype=np.int)
                self.delta_tmp=np.zeros(node_index.size)
                
                self.remove_deltas=np.zeros(node_index.size)#[0 for i in node_index]
                #self.remove_candidates=np.zeros(node_index.size,dtype=np.int)
                self.remove_candidates=[[] for i in node_index]
                #self.rdelta_cache=np.zeros((node_index.size,cache_size))
                #self.rdelta_index=np.zeros((node_index.size,cache_size),dtype=np.int)
                #self.rdelta_tmp=np.zeros(node_index.size)


                for i in self.node_index:
                        self.p_score(i)
                        self.reverse_p_score(i)
                

        """ p_score scores the the realtionship between the parent node and the blanket"""
        def p_score(self,cnode):
                """ Simple first order parent node scoring method"""
                family=[cnode]+self.BN.pnodes[cnode]
                score=self.scores[cnode]
                delta_tmp=self.delta_tmp
                for i in self.BN.p_candidates(cnode):
                        subset=family+[i]
                        new_score=self.objfunc(self.data[:,subset],self.arity[subset])
                        delta_tmp[i]=new_score-score
                self.delta_tmp[delta_tmp<0.0]=0.0
                self.delta_index[cnode,:]=np.argsort(delta_tmp)[-self.cache_size:]
                self.delta_cache[cnode,:]=delta_tmp[self.delta_index[cnode,:]]
                delta_tmp[delta_tmp>0]=0.0

        def reverse_p_score(self,cnode):
                """ Reverse Search that looks for an edge to remove """
                family=[cnode]+self.BN.pnodes[cnode]
                score=self.scores[cnode]
                old_score=score 
                parent=[]
                for i in self.BN.pnodes[cnode]:
                        family.remove(i)
                        new_score=self.objfunc(self.data[:,family],self.arity[family])
                        family.append(i)
                        if score-new_score<=0: score=new_score; parent=[i]
                self.remove_candidates[cnode]=parent
                self.remove_deltas[cnode]=score-old_score



        def add_edge_and_sync(self,cnode,pnode):
                """ Add edge to network, update p-scores, set constraints for edge"""
                #self.BN.p_candidates[cnode].remove(pnode)
                true_pnode=self.delta_index[cnode,pnode]
                self.BN.pconstraints[cnode].add(true_pnode)
                print("adding %d -> %d" %(true_pnode,cnode))
                self.scores[cnode]+=self.delta_cache[cnode,pnode]
                self.delta_cache[cnode,pnode]=0.0
                self.BN.cnodes[true_pnode].append(cnode)
                self.BN.pnodes[cnode].append(true_pnode)
                self.p_score(cnode)
                d_set=self.BN.find_descendants1(cnode)#|set([cnode])
                a_set=self.BN.find_ancestors1(true_pnode)#|set([true_pnode])
                delta_index=self.delta_index
                for i in a_set:
                        #self.deltas[i,[j for j in self.BN.p_candidates[i]&d_set]]=0.0
                        ind=[np.where(delta_index[i,:]==j)[0] for j in set(self.delta_index[i,:])&d_set]
                        self.delta_cache[i,ind]=0.0
                        self.BN.pconstraints[i]|=d_set
                        if not self.delta_cache[self.delta_cache>0.0].size:
                                self.p_score(i)


        def remove_edge_and_sync(self,cnode,pnode):
                """ Remove the edge from BN, relax the constraints and update the p_scores"""
                d_set=self.BN.find_descendants1(cnode)#|set([cnode])
                a_set=self.BN.find_ancestors1(pnode)#|set([pnode])
                self.BN.pnodes[cnode].remove(pnode)
                self.BN.cnodes[pnode].remove(cnode)
                self.BN.pconstraints[cnode]-=set([pnode])
                family=[cnode]+self.BN.pnodes[cnode]
                self.scores[cnode]=self.objfunc(self.data[:,family],self.arity[family])
                self.p_score(cnode)
                for i in self.BN.pnodes[cnode]:
                        a_set-=self.BN.find_ancestors1(i)#|set([i])
                for i in a_set:
                        #family=[i]+self.BN.pnodes[i]

                        #self.BN.pconstraints[i]-=d_set-set([i])
                        self.BN.pconstraints[i]=self.BN.find_descendants(i)|\
                                                                set([i])|set(self.BN.pnodes[i])
                        self.p_score(i)

                        
        def grad_search(self, max_edges=None, tol=10**-6):
                """ 
                Structure learning serach basd on data
               
                max_edges is the upper bound on the number of edges
                added to the BN.

                subroutine for a more sophisticated method.
                """

                # If max_edges is not provided try to add up to three timese as many edges as nodes 
                if max_edges is None: max_edges=3*self.node_index.size


                best_deltas=np.amax(self.delta_cache,axis=1)
                #cnode=np.argmax(best_deltas)
                #pnode=np.argmax(self.delta_cache[cnode,:])
                cur_iter=0
                while max(np.max(best_deltas),np.max(self.remove_deltas))>tol \
                        and cur_iter<max_edges:
                        #print( 'iteration %s' %cur_iter)
                        if max(best_deltas)>max(self.remove_deltas):
                                cnode=np.argmax(best_deltas)
                                pnode=np.argmax(self.delta_cache[cnode,:])
                                #print("adding edge (%d %d) with %f" \
                                #       %(cnode,self.delta_index[cnode,pnode],best_deltas[cnode]))
                                self.add_edge_and_sync(cnode,pnode)
                                self.reverse_p_score(cnode)
                        else:
                                cnode=np.argmax(self.remove_deltas)
                                pnode=self.remove_candidates[cnode][0]
                                #print("removing edge (%d %d) with %f" \
                                #       %(cnode,pnode,self.remove_deltas[cnode]))
                                self.remove_edge_and_sync(cnode,pnode)
                                self.remove_candidates[cnode]=0
                                self.remove_deltas[cnode]=0
                                self.reverse_p_score(cnode)

                        best_deltas=np.amax(self.delta_cache,axis=1)
                        cur_iter+=1

                return self.net_score,np.sum(self.scores)

        def gsrestarts(self,nrestarts=10,tol=10**-6):
                """
                Stochastically perturbed descent search - the primary general searching
                method. 
                
                improves optimality of high order fucntions and avoids local minima
                """
                self.grad_search()
                tmpBN=deepcopy(self.BN)
                tmpscore=np.sum(self.scores)
                for iter in range(nrestarts):
                        [self.BN.remove_random_edge(self.remove_edge_and_sync) for i in \
                        range(np.random.randint(1,self.arity.size/2))]
                        self.score_net()                        
                        self.grad_search()
                        current_score=np.sum(self.scores)
                        if current_score>tmpscore:
                                print('found')
                                tmpBN=deepcopy(self.BN)
                                tmpscore=current_score
                self.BN=deepcopy(tmpBN)
                self.score_net()



                



        def simple_search(self, max_edges=None, tol=10**-6):
                """ Simple data driven structure learning search.
                        max_edges is the upper bound on the number of edges added to the BN.
                """

                # If max_edges is not provided try to add up to twice as many edges as
                # there are nodes 
                if max_edges is None: max_edges=2*self.node_index.size

                for i in self.node_index:
                        self.p_score(i)
                best_deltas=np.amax(self.delta_cache,axis=1)
                cnode=np.argmax(best_deltas)
                pnode=np.argmax(self.delta_cache[cnode,:])
                cur_iter=0
                while np.max(best_deltas)>tol and cur_iter<max_edges:
                        print( 'iteration %s' %cur_iter)
                        print(best_deltas[cnode],cnode,self.delta_index[cnode,pnode])
                        self.add_edge_and_sync(cnode,pnode)

                        best_deltas=np.amax(self.delta_cache,axis=1)
                        cnode=np.argmax(best_deltas)
                        pnode=np.argmax(self.delta_cache[cnode,:])
                        cur_iter+=1
                return self.net_score,np.sum(np.diag(self.scores))


        def score_net(self):
                """ Score the constructed bayseian netowrk """

                score=0
                for child in self.node_index:
                        subset=[child]+self.BN.pnodes[child]
                        self.scores[child]=self.objfunc(self.data[:,subset],self.arity[subset])
                self.net_score=np.sum(self.scores)
                return self.net_score
        

        def score_edge(self, cnode, pnode):
                """ Score the edge given by (cnode,pnode)"""

                family=[cnode]+self.BN.pnodes[cnode]
                score=self.scores[cnode]
                try:
                        family.remove(pnode)
                except ValueError:
                        print(cnode,pnode)
                new_score=self.objfunc(self.data[:,family],self.arity[family])
                return score-new_score

        def stats(self,node=None,filename=''):
                """Write stats to the output file"""
                if not node:
                        nodes=self.BN.node_index
                else: nodes=[node]
                output=''
                for node in nodes:
                        subset=[node]+self.BN.pnodes[node]
                        Nijk,Nij,pstates,H,C=cpt(self.data[:,subset],self.arity[subset])
                        head='\t'.join(['%4s' %val  for val in np.unique(self.data[:,node])])
                        output+='Node: '+str(self.BN.node_names[node])+':'+str(node)+'\n'
                        tmp=''.join(['Ancestors: ',
                                ' '.join(['('+str(self.BN.node_names[i])+':'+str(i)+')'
                                for i in self.BN.pnodes[node]])])
                        output+='\n'.join(tw.wrap(tmp))+'\n'
                        tmp=''.join(['Descendants: ',
                                        ' '.join(['('+str(self.BN.node_names[i])+':'+str(i)+')'
                                for i in self.BN.cnodes[node]])])
                        output+='\n'.join(tw.wrap(tmp))+'\n'
                        output+='H=%f C=%f\n' %(H,C)
                        output+='State\tCount\t'+head+'\n\n'
                        for i,state in enumerate(pstates):
                                output+=''.join([''.join(list(map(str,state))),'\t%4s\t' %(Nij[i]),
                                '\t'.join(['%4s' %(val) for val in Nijk[i,:]]),'\n'])
                        output+='\n\n'
                if filename:
                        print(output,file=open(filename,'w'))
                else:
                        print(output)




        def dot(self, path="", tol=10**-6, cnode=None, radius=None):
                """ Create a dot file for the constructed BN.
                        If cnode is provided create a dot file for Markov neighborhood of
                        that node. 
                """

                BN=self.BN

                if not cnode is None:
                        if radius is None:
                                BN=self.BN.markov_neighborhood(cnode)
                                print( 'Markov neighborhood of %s' %self.BN.node_names[cnode])
                        else:
                                BN=self.BN.subnet_of_radius(cnode,radius)
                                print("Subnet of radius %d around %s" \
                                        %(radius,self.BN.node_names[cnode]) )
                
                # Index map for Markov Net scoring
                ind_map=[self.BN.node_names.index(i) for i in BN.node_names]

                edge_scores=[]
                edges=[]
                
                # Begin the dot string 
                s='digraph G{\n ratio=fill;\n node [shape=box, style=rounded];\n'

                for cnode in BN.node_index:
                        s+='"%s";\n' %BN.node_names[cnode]

                        for pnode in BN.pnodes[cnode]:

                                # In case BN is a Markov Neighborhood translate the edge
                                # indices into their global equivalent and score. Otherwise the
                                # map is an identity
                                cnode_g=ind_map[cnode]
                                pnode_g=ind_map[pnode]
                                edge_score=self.score_edge(cnode_g,pnode_g)
                                if edge_score>tol:
                                        edge_scores.append(edge_score)
                                        edges.append((BN.node_names[pnode],BN.node_names[cnode]))

                
                # Generate grayscale color gradient for all the edges
                color_grad=np.linspace(0.9,0,len(edge_scores))
                indx=np.argsort(edge_scores)
                
                # Add edges and style to the dot string
                for i,j in enumerate(indx):
                        stl='[color="0 0 %s", label="%.2f",style=bold]'\
                                %(color_grad[i],edge_scores[j])
                        s+='"%s" -> "%s" %s;\n' %(edges[j][0],edges[j][1],stl)


                s+='}'
                foo=open(path+'dotfile.dot','w')
                foo.write(s)
                foo.close()
                os.system("dot -Tpdf "+path+"dotfile.dot -o "+path+"outpdf.pdf")
                        
                        
