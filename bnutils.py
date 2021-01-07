#//=============================================================
#// @author rohanahluwalia

#// This code will run the search algorithm to create the markov blanket from an imported csv file
#//=============================================================

import numpy as np
import random
import csv
import os

def bnetload(bnfile):
	""" load the csv file with data """
	lst=[row for row in csv.reader(bnfile)]
	bn=bnet([row[0] for row in lst])
	pnodes=[[int(i) for i in row[1:]] if len(row)>1 else [] for row in lst]
	for c,pset in enumerate(pnodes):
		for p in pset:
			bn.add_edge(c,p)
	return bn


class bnet:
	""" Create the network class"""
	def __init__( self, node_names ):
	# Initialize an empty network

		self.node_names = node_names
		self.bsize=len(node_names)

		self.node_index = np.arange(self.bsize)
		self.pnodes = [[] for i in node_names]
		self.cnodes = [[] for i in node_names]
		#self.p_candidates=[set([i for i in self.node_index if i!=j]) for j in self.node_index]
		self.pcandidates=set(self.node_index)
		self.pconstraints=[set([i]) for i in self.node_index]
		
		self.pmax=self.bsize
		self.required_edges=[]
		self.forbidden_edges=[]

	def __and__(self,net_b):
		"""Determine intersection between nodes"""
		intersection=bnet(self.node_names)
		
		for i,name in enumerate(intersection.node_names):
			[intersection.add_edge1(i,j) for j in\
			set(self.pnodes[self.node_names.index(name)])&\
			set(net_b.pnodes[net_b.node_names.index(name)])]

		return intersection

	def __sub__(self,net_b):
		"""add edges that are indexed"""
		delta=bnet(self.node_names)
		for i,name in enumerate(delta.node_names):
			[delta.add_edge(i,j) for j in\
			set(self.pnodes[self.node_names.index(name)])-\
			set(net_b.pnodes[net_b.node_names.index(name)])]

		return delta

	def p_candidates(self,cnode):
		return self.pcandidates-self.pconstraints[cnode]

	def add_variable(self,name):
		self.node_names.append(name)
		self.bsize+=1
		self.node_index=np.arange(self.bsize)
		self.pnodes.append([])
		self.cnodes.append([])
		self.pconstraints.append(set([self.node_index[-1]]))


	def find_ancestors(self,node):
		"""find the highest relation of the markov blanket"""
		ancestors=set(self.pnodes[node])
		parents=ancestors
		while parents:
			parents_of_parents=set()
			for i in parents: 
				parents_of_parents|=set(self.pnodes[i])-parents
			parents=parents_of_parents
			ancestors|=parents
		return ancestors

	def find_ancestors1(self,node):
		"""determine ancestors without parents"""
		ancestors=set()
		def g(node,ancestors):
			if node not in ancestors:
				ancestors|=set([node])
				for p in self.pnodes[node]:
					g(p,ancestors)
		g(node,ancestors)
		return ancestors

	def find_descendants1(self,node):
		"""determine descendents based on data set"""
		d_set=set()
		def g(node,d_set):
			if node not in d_set:
				d_set|=set([node])
				for c in self.cnodes[node]:
					g(c,d_set)
		g(node,d_set)
		return d_set
	

	def find_descendants(self,node):
		"""descendents based on nodes outputted"""
		descendants=set(self.cnodes[node])
		children=set(self.cnodes[node])
		while children:
			children_of_children=set()
			for i in children:
				children_of_children|=set(self.cnodes[i])-children
			children=children_of_children
			descendants|=children
		return descendants

	def add_edge(self,cnode,pnode):
		"""add edge to network"""
		self.pconstraints[cnode].add(pnode)
		#self.p_candidates[cnode].remove(pnode)
		self.cnodes[pnode].append(cnode)
		self.pnodes[cnode].append(pnode)
		d_set=self.find_descendants1(cnode)|set([cnode])
		a_set=self.find_ancestors1(pnode)|set([pnode])
		for i in a_set:
			#self.p_candidates[i]-=d_set
			self.pconstraints[i]|=d_set

	def add_edge1(self,cnode,pnode):
		"""add edge to network"""
		self.pconstraints[cnode].add(pnode)
		#self.p_candidates[cnode].remove(pnode)
		self.cnodes[pnode].append(cnode)
		self.pnodes[cnode].append(pnode)
		d_set=self.find_descendants1(cnode)
		a_set=self.find_ancestors1(pnode)
		for i in a_set:
			#self.p_candidates[i]-=d_set
			self.pconstraints[i]|=d_set


	def add_random_edge(self):
		"""add random edge so that it can be scored and analyzed"""
		cnode=np.random.randint(0,self.bsize)
		if len(self.pnodes[cnode])<self.pmax and \
			len(self.pconstraints[cnode])<self.bsize:
			#cnode=np.random.randint(0,self.bsize)
			pnode=np.random.choice([i for i in self.p_candidates(cnode)])
			print('found?',pnode in self.pconstraints[cnode])
			self.add_edge(cnode,pnode)
			print('added',self.pconstraints[cnode],self.pconstraints[pnode])
		else:
			print('nonadded')

	def remove_edge(self,cnode,pnode):
		"""remove low-scoring edges"""
		d_set=self.find_descendants(cnode)|set([cnode])
		a_set=self.find_ancestors(pnode)|set([pnode])
		self.pnodes[cnode].remove(pnode)
		self.cnodes[pnode].remove(cnode)
		#self.p_candidates[cnode].add(pnode)
		self.pconstraints[cnode]-=set([pnode])
		for i in a_set:
			self.pconstraints[i]=self.find_descendants(i)|\
								set([i])|set(self.pnodes[i])

	#	for i in self.pnodes[cnode]:
	#		a_set-=self.find_ancestors(i)|set([i])
	#	for i in self.cnodes[pnode]:
	#		d_set-=self.find_descendants(i)|set([i])
	#	for i in a_set:
	#		self.pconstraints[i]-=d_set

	def remove_random_edge(self,remove_function=None):
		"""remove random edge -- this improves accuracy by making the algorithm create new relationships"""
		ind=np.where(np.array([len(i) for i in self.pnodes])>0)[0]
		if ind.size>0:
			cnode=np.random.choice(ind)
			#cnode=np.random.randint(0,self.bsize)
			#if self.pnodes[cnode]:
			pnode=np.random.choice(self.pnodes[cnode])
			print('removing %d -> %d' %(pnode,cnode))
			if remove_function:
				remove_function(cnode,pnode)
			else: 
				self.remove_edge(cnode,pnode)

	def make_random_net(self):
		"""the random net requires the algorithm to create new relationships between target varibles"""
		adj_mat=np.tril(np.random.randint(0,2,size=(self.bsize,self.bsize)),-1)
		self.pnodes=[i.nonzero()[0].tolist() for i in adj_mat]
		self.cnodes=[i.nonzero()[0].tolist() for i in adj_mat.T]
		self.pconstraints=[set(np.arange(i,self.bsize)) for i in range(self.bsize)]
		

	def markov_neighborhood(self,node):
		"""determine the markov neighbhorhood, distance based on lineage"""
		mnodes=[node]+self.pnodes[node]+self.cnodes[node]
		for i in self.cnodes[node]:
			mnodes+=self.pnodes[i]
		mnodes=np.unique(mnodes).tolist()
		markov=bnet([self.node_names[i] for i in mnodes])
		for i,name in enumerate(mnodes):
			for j in set(self.pnodes[name])&set(mnodes):
				markov.add_edge(i,mnodes.index(j))
		return markov

	def subnet_of_radius(self,node,radius=1):
		"""radius of markov blanket --- how long will you branch out""" 
		Mn=self.markov_neighborhood(node)
		print( Mn.node_names)
		mnodes=[self.node_names.index(name) for name in Mn.node_names]
		print (mnodes)
		subnet_nodes=[]
		for r in range(radius):
			for i in mnodes:
				Mn=self.markov_neighborhood(i)
				subnet_nodes+=[self.node_names.index(name) for name in Mn.node_names]
				print (Mn.node_names)
			subnet_nodes=np.unique(subnet_nodes).tolist()
			mnodes=[i for i in subnet_nodes]

		subnet_r=bnet([self.node_names[i] for i in subnet_nodes])
		for i,name in enumerate(subnet_nodes):
			for j in set(self.pnodes[name])&set(subnet_nodes):
				subnet_r.add_edge(i,subnet_nodes.index(j))

		return subnet_r

	def dot(self):
		"""write out to dot file with results (markov blanket...)"""
		s='digraph G{\n ratio=fill;\n'

		for child in self.node_index:
			s+='"%s";\n' %self.node_names[child]
			for parent in self.pnodes[child]:
				s+='"%s" -> "%s";\n' %(self.node_names[parent],self.node_names[child])
		s+='}'
		dotfile=open('dotfile.dot','w')
		dotfile.write(s)
		dotfile.close()
		os.system("dot -Tpdf dotfile.dot -o outpdf.pdf")

	"""save completed bayesian network file"""
	def bnetsave(self,filename='bnstruct.csv'):
		fout=open(filename,'w')
		csvwr=csv.writer(fout)
		for i,name in enumerate(self.node_names):
			csvwr.writerow([name]+self.pnodes[i])
		fout.close()


