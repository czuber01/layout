# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:58:14 2017

@author: Charlotte
"""

import numpy as np
import networkx as nx
import random
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as sio
import pandas as pd

mat_contents=sio.loadmat('TFgraph.mat')
adjmat_disease=mat_contents['PATH']
graph_disease=nx.from_numpy_matrix(adjmat_disease)
#G = nx.binomial_graph(1000,random.random())
G=nx.erdos_renyi_graph(50,0.1)
#start with a basic layout
#nodesarray=[]
#def coarsengraphrec(graph,pos,threshold):
#    if (graph==threshold):
#        nodesarray.push(??)
#        return
#    graph2,pos2,nodes=coarsengraph(graph,pos)
#    coarsengraphrec(graph2,pos2,threshold)
#    nodesarray.push(nodes)
#
#def coarsengraph(graph,pos):
#    # create list of unmatched vertices
#    
#    pos2[loc]=pos[a]/2+pos[b]/2 #put into vector form
#    return graph2,pos2,nodes
#    
#    

#nodes array set up so that each position with two objects is a true child node??
#Gnext=GL
#for i in range(L):
#    k=k*function
#    R=R*function
#    #outputs a 2d position vector
#    Gnext,pos=mergegraph(Gnext,pos,nodes[L-i])
#    pos=layout(Gnext,pos,k,R)


#def mergegraph(G,pos,nodes):
#    #take graph/positions and replace nodes with their parent nodes in roughly same spot
#    G2=[]
#    pos2=[[2*pos.length]]
#    loc=0
#    for i in G:
#        if nodes[i].length==1:
#            G2[loc]=G[i]
#            pos2[loc]=pos[i]
#            loc +=1
#        else:
#            G2[loc]=nodes[i][0]
#            G2[loc+1]=nodes[i][1]
#            pos2[loc]=pos[i]
#            pos2[loc+1]=pos[i]
#            loc+=2
#    return G2,pos2

# send in graph with fixed positions, a k(optdist) value, and an R value
#is the matrix considered sparse enough to use the simplification? if E ~ V
#def layout(G,pos_array,k,R):
def layout(G,iter,repulstype,repulsfactor,centralityfraction):
    adjmat=nx.adjacency_matrix(G)
    adjmat=adjmat.todense()
    a,b=adjmat.shape
    adjmat=np.asarray((adjmat),dtype='float64')
    iterations=iter
    pos_array=np.random.RandomState(42)
    pos=pos_array.rand(a,2)
    #pos=pos_array
    pos=pos.astype(adjmat.dtype)
    k=np.sqrt(1.0/a)
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1]))*0.1
#    dt=t/float(iterations+1)
    dt=0.9   
    delta = np.zeros((pos.shape[0],pos.shape[0],pos.shape[1]),dtype=adjmat.dtype)
    degree=nx.degree_centrality(G)
    betweenness=nx.betweenness_centrality(G)
    betweenness=np.array(betweenness.values(),dtype='float64')*(1-centralityfraction)
    degree=np.array(degree.values(),dtype='float64')*centralityfraction
    degreemat=[[i*j/centralityfraction for i in degree]for j in degree]
    betweenmat=[[i*j/(1-centralityfraction) for i in betweenness] for j in betweenness]
    centralitymat=[[x+y for x,y in zip(degreemat[i],betweenmat[i])]for i in range(len(betweenmat))]
    energy=[]
    progress=0
    for iteration in range(iterations):
        for i in range(pos.shape[1]):
            #each 2d array is for one coordinate
            delta[:,:,i]=pos[:,i,None]-pos[:,i]
        delta=np.array(delta,dtype=np.float64)
        #if magnitude of delta<R(k), global force=0
        distance=np.sqrt((delta**2).sum(axis=-1))
        distance=np.where(distance<0.01,0.01,distance)
        #repulsion function, edge force not dependent on 
        #weight of the edge, just node to node interaction
        if repulstype =='edgespring1':
            edgeforce=degree + betweenness
            edgeforce=edgeforce/distance**2
            # old version used in jupyter edgeforce=edgeforce/edgefactor
            springrepuls=k*k*k/distance**2
            edgerepuls=edgeforce/(repulsfactor)
            force1=springrepuls+edgerepuls
            force1=force1/repulsfactor

        if repulstype == 'edgespring2':           
            force_edge=centralitymat
            force_edge=force_edge/(distance**2)
            # old version used in jupyter force1=force_edge/edgefactor+(k*k/distance**2)/repulsfactor
            force1=force_edge/repulsfactor/10+(k*k*k/distance**2)/repulsfactor
        
        if repulstype == 'trial':
            force_edge=centralitymat
            force_edge=force_edge/(distance**2)
            # old version used by jupyter force1=force_edge/edgefactor
            force1=force_edge/repulsfactor
        #follows linlog documentation
        if repulstype == 'linlog':
            force_edge=centralitymat
            force_edge=force_edge*np.log(distance)
            force1=-1*force_edge
        if repulstype == 'noedge':
            force1=k*k/distance**2
            force1*=1/repulsfactor
        if repulstype == 'spring':
            force1=k*k/distance
            force1*=1/repulsfactor
        #attraction function
        force2=adjmat*distance/k
        forces=force1-force2
        displacement1=np.transpose(np.transpose(delta)*(forces))
        displacement=displacement1.sum(axis=1)
        #displacement is a 
        length=np.sqrt((displacement**2).sum(axis=1))
        # following eq in http://yifanhu.net/PUB/graph_draw_small.pdf page 5
        energy.append(length.sum(axis=0))
        length=np.where(length<0.01,0.01,length)
        delta_pos=np.transpose(np.transpose(displacement)*t/length)
        pos+=delta_pos
        if (iteration>0 and energy[iteration]<energy[iteration-1]):
            progress+=1
            if progress>=5:
                progress=0
                t=t/dt
        else: 
            progress=0
            t*=dt                            
#        t-=dt
        #for no fixed nodes only
        pos=_rescale_layout(pos)
        #if delta_pos<tolerance
    plt.plot(energy[5:])
    return pos
    # return  coords dict with id original as key 
        
def _rescale_layout(pos,scale=1.):
    maxlim=0
    for i in range(pos.shape[1]):
        pos[:,i]-= pos[:,i].min()
        maxlim=max(maxlim,pos[:,i].max())
    if maxlim>0:
        for i in range(pos.shape[1]):
            pos[:,i]*=scale/maxlim
    return pos

#nx.draw(G)
#L=layout(graph_disease,50,'trial',10)
L2=layout(graph_disease,50,'edgespring1',100,.99 )
L3=layout(graph_disease,50,'edgespring1',100,.01)
L4=layout(graph_disease,50,'edgespring1',100,0.5)

#L2=layout(G,50)

#plt.figure('springlayout')
#nx.draw_networkx(graph_disease)
#plt.figure('trial1 repuls=10')
#nx.draw_networkx(graph_disease,pos=L)
#plt.figure('linlog')
#nx.draw_networkx(graph_disease,pos=L2)
plt.figure('edgespring1 repuls=100 frac=.01')
nx.draw_networkx(graph_disease,pos=L3)
plt.figure('edgespring1 repuls=100 frac=0.5')
nx.draw_networkx(graph_disease,pos=L4)
plt.figure('edgespr1 repuls=100 grac=.99')
nx.draw_networkx(graph_disease,pos=L2)

#
plt.show()  