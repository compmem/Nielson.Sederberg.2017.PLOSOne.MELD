
# coding: utf-8

# In[1]:

import os
import sys
import time
import numpy as np
import nibabel as nib
#import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import nibabel as nib
#import process_log as pl
import pandas as pd
import cPickle as pickle
from joblib import Parallel,delayed

# Connect to an R session
#import rpy2.robjects
#r = rpy2.robjects.r

# For a Pythonic interface to R
#from rpy2.robjects.packages import importr
#from rpy2.robjects import Formula, FactorVector
#from rpy2.robjects.environments import Environment
#from rpy2.robjects.vectors import DataFrame, Vector, FloatVector
#from rpy2.rinterface import MissingArg,SexpVector

# Make it so we can send numpy arrays to R
#import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()
#fdrtool = importr('fdrtool')
#%load_ext rmagic



# In[2]:

import ptsa.stats.meld as meld
from ptsa.stats import cluster as cl
from ptsa.stats.nonparam import gen_perms


# #Defining functions

# In[3]:

def sig_by_prop(p,size=(100,100)):
    field = np.zeros(size)
    n = (field.shape[0]*field.shape[1])*p
    randx= np.random.randint(0,field.shape[0],n)
    randy= np.random.randint(0,field.shape[1],n)

    field[randx,randy] = 1
    return field

def hide_blobs(nfeats=(100,100), x=0, y=0,blob_pat='big',zfield = None,cfrac = None):
    """ puts a (artisinally hand crafted) set of 1's surrounded by 0's
        into a field of dims nfeats
        currently supported patterns are small, final, big, vbig,ten,hundo,
        for the MELD paper, I've added central, split, and dispersed, 
        these hijack your x and y and assume you've got 100 features""" 

    #check input
    if x > nfeats[0] or y > nfeats[1]:
        print "You tried to place the blobs outside the indicies of the feature space you asked for."
        print "setting x and y to 0"
        x = y = 0
    
    if zfield is None:
        zfield = np.zeros(nfeats)
    
    if cfrac != None:
    #make center blob
        facs = np.sort(list(factors(cfrac)))
        if len(facs)%2==0:
            cdim = facs[len(facs)//2-1:len(facs)//2+1]
        else:
            cdim = np.array([facs[len(facs)//2],facs[len(facs)//2]])
        #figure out how to center center blob
        cyx = (np.array(nfeat)//2)-(cdim//2)

        #hide center
        zfield[cyx[0]:cyx[0]+cdim[0],cyx[1]:cyx[1]+cdim[1]] = 1

        dfrac = 100-cfrac

        dyxlist = [10,30,50,70,90]

        dsum = 0
        for y in dyxlist:
            for x in dyxlist:
                if (x != 50) | (y!=50):
                    if dfrac-dsum>=4:
                        ddim = np.array([2,2])
                    elif dfrac-dsum>0:
                        ddim = np.array([dfrac-dsum,1])
                    else:
                        break
                    dsum += np.ones(ddim).sum()
                    zfield[y:y+ddim[0],x:x+ddim[1]]=1
        return zfield
    
    xs = None
    #This is what our signal will eventually be multiplied by to give a spatially distributed signal
    
    if blob_pat =='small':
        blobs = np.array ([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,1.,1.,0.,0.,1.,1.,0.,0.,1.,0.,0.,0.,1.],
                           [0.,0.,0.,1.,1.,0.,0.,1.,1.,0.,0.,1.,1.,0.,0.,1.],
                           [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,1.,1.,0.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,1.,1.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,0.],
                           [-1.,0.,0.,-1.,-1.,0.,0.,-1.,-1.,0.,0.,-1.,-1.,-1.,-1.,-1.],
                           [-1.,0.,0.,0.,-1.,0.,0.,-1.,-1.,0.,0.,0.,-1.,-1.,-1.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.]])
        
    elif blob_pat =='final':
        blobs = np.array ([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],
                            [1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                            [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0.166666667,0.166666667,0,0,0.333333333,0.333333333,0,0,0.5,0.5,0,0,0.666666667,0.666666667,0,0,0.833333333,0.833333333,0,0,1,1],
                            [0.166666667,0.166666667,0,0,0.333333333,0.333333333,0,0,0.5,0.5,0,0,0.666666667,0.666666667,0,0,0.833333333,0.833333333,0,0,1,1],
                            [0.166666667,0.166666667,0,0,0.333333333,0.333333333,0,0,0.5,0.5,0,0,0.666666667,0.666666667,0,0,0.833333333,0.833333333,0,0,1,1],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [-0.166666667,-0.166666667,0,0,-0.333333333,-0.333333333,0,0,-0.5,-0.5,0,0,-0.666666667,-0.666666667,0,0,-0.833333333,-0.833333333,0,0,-1,-1],
                            [-0.166666667,-0.166666667,0,0,-0.333333333,-0.333333333,0,0,-0.5,-0.5,0,0,-0.666666667,-0.666666667,0,0,-0.833333333,-0.833333333,0,0,-1,-1],
                            [-0.166666667,-0.166666667,0,0,-0.333333333,-0.333333333,0,0,-0.5,-0.5,0,0,-0.666666667,-0.666666667,0,0,-0.833333333,-0.833333333,0,0,-1,-1],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,-1],
                            [-1,0,0,0,-1,-1,0,0,-1,-1,0,0,-1,-1,0,0,-1,-1,0,0,-1,-1],
                            [0,0,0,0,0,0,0,0,0,-1,0,0,-1,-1,0,0,-1,-1,0,0,-1,-1]])
    elif blob_pat =='vbig':
        blobs = np.array ([[1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,-1.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,-1.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,-1.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,-1.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,-1.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,-1.]])
        
    elif blob_pat =='ten':
        blobs = np.array ([[0.,1.,1.,0.],
                           [0.,1.,1.,1.],
                           [0.,1.,1.,1.],
                           [0.,1.,1.,0.]])
        
    elif blob_pat =='hundo':
        blobs = np.array ([[0.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [0.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [0.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [0.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [0.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [0.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,0.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,0.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,0.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,0.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,0.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,0.]])
    elif blob_pat == 'central01':
        x = 49
        y = 47
        blobs = np.ones((5,2))
    
    elif blob_pat == 'central':
        x = y = 45
        blobs = np.ones((10,10))
        
    elif blob_pat == 'central10':
        x = 37
        y = 30
        blobs = np.ones((25,40))
        
    elif blob_pat == 'split01':
        xs = [40,60]
        ys = [47]
        blobs = np.ones((5,1))

    elif blob_pat == 'split':
        xs = ys = [40,60]
        blobs = np.ones((5,5))
        
    elif blob_pat == 'split10':
        xs = [15,60]
        ys = [30,60]
        blobs = np.ones((10,25))
    
    elif blob_pat == 'dispersed01':
        xs = [40,60]
        ys = [10,30,50,70,90]
        blobs = np.ones((1,1))

    elif blob_pat == 'dispersed':
        xs = ys = [10,30,50,70,90]
        blobs = np.ones((2,2))
    
    elif blob_pat == 'dispersed10':
        xs = ys = [6,26,46,66,86]
        blobs = np.ones((5,8))

    else:
        if blob_pat != 'big':
            print "I don't know that shape, setting shape to big"
        blobs = np.array ([[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.],
                           [1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.],
                           [0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,0.],
                           [0.,0.,-1.,-1.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.],
                           [0.,0.,-1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.]])
    #this sets blobs equal to the dims of nfeat-placement indices
    if nfeats[0] - x < np.shape(blobs)[0]:  
        blobs = blobs[0:nfeats[0]-x,:]
    if nfeats[1] - y  < np.shape(blobs)[1]:
        blobs = blobs[:,0:nfeats[1]-y]
        
        
    if xs is not None:
        for y in ys:
            for x in xs:
                zfield[y:y+blobs.shape[0],x:x+blobs.shape[1]] = blobs

    else:
        zfield[x:x+blobs.shape[0],y:y+blobs.shape[1]] = blobs
    return zfield


# In[4]:

def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


# In[5]:

def gen_data(nsubj,nobs,nfeat,slope,sigfield,mnoise=False,contvar=False,item_inds=False,mod_cont=None,
             I = 0.0, S = 0.0 ):
    """Generate a set of random data with a independent data and dependent data matricies
       Data will have nsubj*nobs rows and dep data will have 1 column for each feature
       sigfield is the matrix the signal should be multiplied by to give it a "spatial" extent
       Independent by default has a factor called beh which represents some behavioral condition
       if mnoise,  then the slopes have subject and item level noise added to them
                ie, this is the type of noise you are modeling with (x|subj) or (x|item)
       if contvar, then there is a continuous variable in the data that interacts with the
                beh factor
       if mod_cont, then the continuous variable is included in the output independent data
                This is here incase I want to look at the effects of including a continuous
                variable in your data when its not there and vice versa
        ##As of 3-7-14, item level noise is used only for slopes.##
       if item_inds is true then the independent variable table is output with a column for item
                item level noise is used to generate the model regardless of this option
                """
    
    if mod_cont is None:
        mod_cont = contvar
    
    #talked about implementing blur, but haven't
    #blur = 2
    #define slopes, they are all set to slope for simplicity for now
    #could set it to accept a dict of slopes or some such
    beh_slope = slope
    cont_slope = slope
    bxc_slope = slope
    
    #set up the behavioral array, this is set up for two behavioral conditions currently
    beh_list = np.array([-0.5,0.5]*(nobs/2))
    if nobs != len(beh_list):
        nobs = len(beh_list)
        print 'The number of observations you provided was not divisible by 2,'
        print 'it has been set to', nobs
        
    #generate independent variables and put it in a rec_array
    s = np.concatenate([np.array([i]*nobs) for i in range(nsubj)])
    beh = np.reshape(np.array([np.random.permutation(beh_list) for i in range(nsubj)]),np.shape(s))
    items = np.reshape(np.array([np.random.permutation(nobs) for i in range(nsubj)]),np.shape(s))
    
    ind_data = np.rec.fromarrays((np.random.randn(len(s)),
                                  np.random.randn(len(s)),
                                  beh,
                                  items,
                                  s),
                                  names='val,cont,beh,item,subj')
    
    #set up noise, noise is all random standard normal
    #noise has 6 columns, 2 for each of beh, cont, bxc
    snoise = np.random.standard_normal((nsubj,6))*S
    itemnoise = np.random.standard_normal((nobs,6))*I
    #n_denom = 1+inoise
    #noisy field to which the signal y will be added
    dep_data = np.random.randn(len(s),*nfeat)
    
    
    #save out stats on noisy field before signal is added to it
    sn_stats = dict(noise=dict(ave=np.mean(dep_data[:,sigfield!=0]),
                               sigma=np.std(dep_data[:,sigfield!=0]),
                               high=np.max(dep_data[:,sigfield!=0]),
                               low=np.min(dep_data[:,sigfield!=0])))
   
    ys = np.zeros((len(ind_data),sigfield.shape[0],sigfield.shape[1]))
    
    #loop through ind_data to created corresponding dep_data for each row
    for i in range(len(ind_data)):
        
        if mnoise == True:
            #mx is the ind data * slope * average
            #with slope noise
            #Previous methods of generating slope
            #beh_mx = ind_data['beh'][i]*beh_slope*((snoise[ind_data['subj'][i],1]+itemnoise[ind_data['subj'][i],1])/n_denom)
            #cont_mx = ind_data['cont'][i]*cont_slope*((snoise[ind_data['subj'][i],3]+itemnoise[ind_data['subj'][i],3])/n_denom)
            #beh_mx = ind_data['beh'][i]*(beh_slope+((snoise[ind_data['subj'][i],1]+itemnoise[ind_data['subj'][i],1])/n_denom))
            #cont_mx = ind_data['cont'][i]*(cont_slope+((snoise[ind_data['subj'][i],3]+itemnoise[ind_data['subj'][i],3])/n_denom))
            
            #beh_mx = ind_data['beh'][i]*(beh_slope+(snoise[ind_data['subj'][i],1]+itemnoise[ind_data['subj'][i],1]))
            #cont_mx = ind_data['cont'][i]*(cont_slope+(snoise[ind_data['subj'][i],3]+itemnoise[ind_data['subj'][i],3]))
            #bxc_mx = ind_data['beh'][i]*ind_data['cont'][i]*(beh_slope*cont_slope+(snoise[ind_data['subj'][i],5]+itemnoise[ind_data['subj'][i],5]))
            
            #Took out the item level slope noise after talking with Per on 3-7-14
            beh_mx = ind_data['beh'][i]*(beh_slope+(snoise[ind_data['subj'][i],1]))
            cont_mx = ind_data['cont'][i]*(cont_slope+(snoise[ind_data['subj'][i],3]))
            bxc_mx = ind_data['beh'][i]*ind_data['cont'][i]*(beh_slope*cont_slope+(snoise[ind_data['subj'][i],5]))
        else:
            #no  slope noise
            beh_mx = ind_data['beh'][i]*beh_slope
            cont_mx = ind_data['cont'][i]*cont_slope
            bxc_mx = ind_data['beh'][i]*ind_data['cont'][i]*bxc_slope
        
        #b is the average of item and subject intercept noise, Old methods
        #beh_b = (snoise[ind_data['subj'][i],0] + itemnoise[ind_data['subj'][i],0])/n_denom
        #cont_b = (snoise[ind_data['subj'][i],2] +itemnoise[ind_data['subj'][i],2])/n_denom        
        #beh_b = (snoise[ind_data['subj'][i],0] + itemnoise[ind_data['subj'][i],0])
        #cont_b = (snoise[ind_data['subj'][i],2] +itemnoise[ind_data['subj'][i],2])
        #bxc_b = (snoise[ind_data['subj'][i],4] +itemnoise[ind_data['subj'][i],4])
        
        #Took out the item level intercept noise after talking with Per on 3-7-14
        #Put it back in after looking at http://talklab.psy.gla.ac.uk/KeepItMaximalR2.pdf
        #page 13
        beh_b = (snoise[ind_data['subj'][i],0] + itemnoise[ind_data['subj'][i],0])
        cont_b = (snoise[ind_data['subj'][i],2] +itemnoise[ind_data['subj'][i],2])
        bxc_b = (snoise[ind_data['subj'][i],4] +itemnoise[ind_data['subj'][i],4])
        
        #set up the signal for the ith row of ind data
        if contvar == True:
            ys[i,:,:] = sigfield * (beh_mx + beh_b + cont_mx + cont_b + bxc_mx + bxc_b)
        else:
            ys[i,:,:] = sigfield * (beh_mx + beh_b)
        
        #add signal to noisy field
        dep_data[i,:,:] = dep_data[i,:,:] + (ys[i,:,:])
    
    #save stats on signal before it is added to noise
    sn_stats['ys']=dict(ave=np.mean(abs(ys[:,sigfield!=0])),
                        sigma=np.std(abs(ys[:,sigfield!=0])),
                        high=np.max(abs(ys[:,sigfield!=0])),
                        low=np.min(abs(ys[:,sigfield!=0])))
    #save stats on combined signal and noise
    sn_stats['S_N']=dict(ave=np.mean(dep_data[:,sigfield!=0]),
                        sigma=np.std(dep_data[:,sigfield!=0]),
                        high=np.max(dep_data[:,sigfield!=0]),
                        low=np.min(dep_data[:,sigfield!=0]))
    
    #modify the ind_data table as necessary for output
    if item_inds == True:
        if mod_cont == True:
            ind_data = ind_data
        else:
            ind_data = ind_data[['val','beh','subj','item']]
    else:
        if mod_cont == True:
            ind_data = ind_data[['val','beh','subj','cont']]
        else:
            ind_data = ind_data[['val','beh','subj']]
    
    #data = (ind_data,ndimage.gaussian_filter(dep_data,2))
    data = (ind_data,dep_data,sn_stats)
    return data


# In[6]:

def get_error_rates(bThr,brs,signal,terms,pvals,full=False):
    """calculate error rate at a single test statistic threshold (bThr)"""
    if full == True:
        pvalThr=0.05
    else:
        pvalThr = 1-bThr
    no_signal = abs(abs(signal) -1.0)
    tp = 0.
    fp = 0.
    fn = 0.
    tn = 0.
    #set postitive and negative to the the number of signal or noise voxels * the number of terms
    if 'Intercept' in terms or '(Intercept)' in terms:        
        p = (len(terms)-1) * np.count_nonzero(signal)
        n = (len(terms)-1) * np.count_nonzero(no_signal)
    else:
        p = (len(terms)) * np.count_nonzero(signal)
        n = (len(terms)) * np.count_nonzero(no_signal)
    for i in range(len(terms)):
        #meld calls the intercept (Intercept), GLM doesn't use parentheses
        if terms[i] != '(Intercept)' and terms[i] != 'Intercept' :
            if pvals[terms[i]] <= pvalThr :
                tp += np.count_nonzero(((abs(brs[terms[i]])>=bThr))*signal)
                fp += np.count_nonzero(((abs(brs[terms[i]])>=bThr))*no_signal)
                if full == True:
                    fn += np.count_nonzero(((abs(brs[terms[i]])<bThr))*signal)
                    tn += np.count_nonzero(((abs(brs[terms[i]])<bThr))*no_signal)
            elif full == True:
                fn += np.count_nonzero(signal)
                tn += np.count_nonzero(no_signal)
    #true positive rate = true positives found divided by potential number of true positives
    try:
        tpr = tp / p
    except:
        tpr = np.NaN
    #false negative rate = false negatives divided by potential number of false negatives
    try:
        fnr = fn / p
    except:
        fnr = np.NaN
    #false positive rate = false positives found divided by potential number of false positives
    fpr = fp / n
    tnr = tn / n
    if full == True:
        return(tpr,fpr,tnr,fnr)
    else:
        return (tpr,fpr)

#new error rate function that just gets error rate for the map it's passed, 
#doesn't care about terms
def get_error_terms(bmap,bsig):
    tp = np.float64(np.sum(bmap*bsig))
    fp = np.float64(np.sum(bmap*~bsig))
    fn = np.float64(np.sum(~bmap*bsig))
    tn = np.float64(np.sum(~bmap*~bsig))
    mccD =(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    if mccD == 0.:
        mccD = 1.
    else:
        mccD = np.sqrt(mccD)
    
    mcc = ((tp*tn)-(fp*fn))/mccD
    
    error_terms = np.array([(tp, 
                             fp,
                             fn,
                             tn,
                             tp/(tp+fn),
                             tn/(fp+tn),
                             tp/(tp+fp),
                             tn/(tn+fn),
                             fp/(fp+tn),
                             fp/(fp+tp),
                             fn/(fn+tp),
                             (tp+tn)/((tp+fn)+(fp+tn)),
                             (2*tp)/(2*tp+fp+fn),
                             mcc)],
                           dtype=np.dtype([('tp', 'float64'),
                                           ('fp', 'float64'),
                                           ('tn', 'float64'),
                                           ('fn', 'float64'),
                                           ('tpr', 'float64'),
                                           ('spc', 'float64'),
                                           ('ppv', 'float64'),
                                           ('npv', 'float64'),
                                           ('fpr', 'float64'),
                                           ('fdr', 'float64'),
                                           ('fnr', 'float64'),
                                           ('acc', 'float64'),
                                           ('f1', 'float64'),
                                           ('mcc', 'float64')]))

    return error_terms

def get_ROC(statmap,signal,start=0.,stop=1.,num=1000,thr=None, ret_thr = False,ret_mcc = False):
    """evaluate true and false positive rates for 1000 values from [0:1],
       calcualtes area under tpr vs fpr (ROC) curve based on trapezoidal method,
       expects positive results to be greater than threshold"""
    if thr is None:
        thr = np.linspace(start,stop,num)
        thr = np.append(thr,thr[-1]+thr[1])

    tprs = np.zeros(len(thr))
    fprs = np.zeros(len(thr))
    mcc = np.zeros(len(thr))
    for i in range(len(thr)):
        #get error rates for each bThr
        et=get_error_terms(statmap>=thr[i],abs(signal)>0)
        tprs[i] = et['tpr']
        fprs[i] = et['fpr']
        mcc[i] = et['mcc']
    data = np.array(sorted(zip(fprs,tprs)))
    #find area under curve using trapezoidal method
    under = np.trapz(data[:,1],data[:,0])
    if ((ret_thr == True)&(ret_mcc==True)):
        return (under, tprs,fprs,thr,mcc)
    elif ret_thr == True:
        return (under, tprs,fprs,thr)
    elif ret_mcc == True:
        return (under, tprs,fprs,mcc)
    else:
        return (under,tprs,fprs)


def eval_bthrs(brs,signal,terms,pvals):
    """evaluate true and false positive rates for a variety of thresholds,
       calcualtes area under tpr vs fpr (ROC) curve based on trapezoidal method"""
    bThr = []
    for t in terms:
        if t != '(Intercept)' and t != 'Intercept' :
            bThr = np.append(bThr,brs[t].flatten())
    bThr = np.unique(bThr)
    tprs = np.zeros(len(bThr))
    fprs = np.zeros(len(bThr))
    for i in range(len(bThr)):
        #get error rates for each bThr
        (tprs[i],fprs[i])=get_error_rates(bThr[i],brs,signal,terms,pvals)
    data = np.array(sorted(zip(fprs,tprs)))
    #find area under curve using trapezoidal method
    under = np.trapz(data[:,1],data[:,0])
    return (under,tprs,fprs)


# In[7]:

def eval_glm(fe_formula,ind_data,dep_data):
    """evaluate one subject's worth of GLM data at each feature"""
    betas = {}
    #get shape of data
    nfeat = np.shape(dep_data[0])
    #iterate over each feature in data
    for p in range(nfeat[0]):
        for q in range(nfeat[1]):
            #set val in ind_data equal to dep_data for that feature
            ind_data['val']=dep_data[:,p,q]
            #fit glm based on model given in fe_formula
            modl= smf.glm(formula=fe_formula, data=ind_data).fit()
            #save beta's for each factor to a dict
            for fac in modl.pvalues.keys():
                if fac not in betas:
                    betas[fac]=np.zeros(nfeat)                
                betas[fac][p,q]=modl.params[fac]
    return betas


# In[8]:

#execution time: 
#size,    pthresh, real,    user
#100x100, p= 0.05, 63.999s, 463.004s
#100x100, p= 0.01, 32.149s, 249.616s
#100x100, p=0.10, 101.395s, 729.352s
#set up alphasim lists
#alphasim value for clusters of index+1 voxels in size
asims_dd={'32x32':{'0.01':[0.999974,
               0.175825,
               0.005355,
               0.000164,
               0.000007,
               0.000001],
       '0.05':[0.999999,
               0.987283,
               0.444893,
               0.081238,
               0.012722,
               0.002005,
               0.000322,
               0.00005,
               0.000008],
       '0.1':[0.999999,
              0.999999,
              0.979591,
              0.64708,
              0.255378,
              0.083128,
              0.025962,
              0.008131,
              0.002557,
              0.000791,
              0.000238,
              0.000072,
              0.000017,
              0.000007,
              0.000003]},
       '100x100':{'0.01':[1.000000,
                          0.853355,
                          0.054625,
                          0.001695,
                          0.000040],
                  '0.05':[1.000000,
                          1.000000,
                          0.997517,
                          0.585414,
                          0.127460,
                          0.021857,
                          0.003581,
                          0.000600,
                          0.000106,
                          0.000011,
                          0.000004,
                          0.000001],
                  '0.10':[1.000000,
                           1.000000,
                           1.000000,
                           0.999983,
                           0.955854,
                           0.607394,
                           0.251411,
                           0.086920,
                           0.028865,
                           0.009338,
                           0.003028,
                           0.000993,
                           0.000328,
                           0.000114,
                           0.000043,
                           0.000013,
                           0.000003,
                           0.000001,
                           0.000001]}}


# In[9]:

def get_alphamap(tvals,pvals,pthr,makez=False,asims=asims_dd['32x32']):
    """code for correction of glm pvals based on frequency of finding
    a cluster of a given size in a field of randomly generated noise.
    Data generated useing Alphasim with 1000000 simulations with no smoothing."""
    #pthr should be equal to 0.1,0.05, or 0.01

    clustmap={}
    numclust={}
    #things get weird in here dealing with p-vals and positive and negative tvals
    #my solution for positive and negative is to cluster them separately and combine them
    #    
    clustmap['pos'],numclust['pos']=ndimage.measurements.label(np.logical_and(pvals<=pthr,tvals > 0))
    clustmap['neg'],numclust['neg']=ndimage.measurements.label(np.logical_and(pvals<=pthr,tvals < 0))
    minp=1
    alphazmap= {k:clustmap[k].astype(np.float) for k in clustmap.keys()}
    for tsign in clustmap.keys():
        for numvox in range(1,numclust[tsign]+1):
            if np.sum(clustmap[tsign]==numvox) > len(asims[str(pthr)]):
                if makez == True:
                    alphazmap[tsign][clustmap[tsign]==numvox]=stats.norm.ppf(asims[str(pthr)][-1]/2,0,1)
                else:
                    alphazmap[tsign][clustmap[tsign]==numvox]=asims[str(pthr)][-1]
                minp=asims[str(pthr)][-1]
            else:
                if makez==True:
                    alphazmap[tsign][clustmap[tsign]==numvox]=stats.norm.ppf(asims[str(pthr)][np.sum(clustmap[tsign]==numvox)-1]/2,0,1)
                else:
                    alphazmap[tsign][clustmap[tsign]==numvox]=asims[str(pthr)][np.sum(clustmap[tsign]==numvox)-1]
                if asims[str(pthr)][np.sum(clustmap[tsign]==numvox)-1] < minp:
                    minp = asims[str(pthr)][np.sum(clustmap[tsign]==numvox)-1]
        
    
    #The p-val returrned is the minimum alphasim corrected pvalue for any cluster
    if makez == True:
        alphamap = alphazmap['neg'] - alphazmap['pos']
    else:
        alphamap=alphazmap['neg'] + alphazmap['pos']
        alphamap[alphamap==0]=1
    #out = dict(alphamap=alphamap,
    #           pval=minp)
    return alphamap


# In[10]:

def get_fdrmap(ts,ret ='qval',stat='normal',center=True):
    tshape = ts.shape
    if center == True:
        fachist = np.histogram(ts,bins=500)
        peak = fachist[1][fachist[0]==np.max(fachist[0])][0]
        ts -= peak
    ts = FloatVector(ts.flatten())
    results = fdrtool.fdrtool(ts,statistic=stat,
                              plot=False, verbose=False)
    return np.array(results.rx(ret)).reshape(tshape)


# In[11]:

#functions needed for ttest
# calc stats for each perm
def eval_perm(perm,ind_data,dep_data):   
    # get the permed ind_data
    pdat = ind_data[perm]
    
    # get the condition indicies based on the permed ind_data
    behA_ind =pdat['beh']==-0.5
    behB_ind = pdat['beh']==0.5
    
    # loop over subjects and calc diff
    vals = []
    for s in np.unique(list(pdat['subj'])):
        # get the subject index
        subj_ind = (pdat['subj']==s)
        
        # get diff in means for that index
        valdiff = (np.average(dep_data[behA_ind&subj_ind],axis=0)-np.average(dep_data[behB_ind&subj_ind],axis=0))
        
        vals.append(valdiff)
    vals = np.array(vals)
    
    # perform stats across subjects
    t,p = stats.ttest_1samp(vals, 0.0, axis=0)
    
    return t[np.newaxis,:]

def run_tfce(t,con,dt=0.05,E=2/3.,H=2.0):
    # enhance clusters in both directions, then combine
    ttemp = cl.tfce(t, tail=1,connectivity=con,dt=dt,E=E, H=H,)
    ttemp += cl.tfce(t, tail=-1, connectivity=con,dt=dt,E=E, H=H,)
    # prepend axis (for easy concatenation) and return
    return ttemp[np.newaxis,:]


# In[22]:

def test_sim_dat(nsubj,nobs,slope,signal,mnoise=False,contvar=False,item_inds=False,
                 mod_mnoise=False,mod_cont=False,use_ranks=False,nfeat=(32,32),save_lfud=False,
                 n_jobs = 2,pthr=0.05,save_maps=False,nperms=200,nboots=50,ntperms=500,save_ROC=False,I=0.0,S=0.0,
                 feat_nboot=1000,feat_thresh=0.1,do_tfce=True,do_ttfce=True,connectivity=None,asims_dd=asims_dd,dt=0.05,E=2/3.,
                 H=2.0,data = None):
    """Executes MELD and ttest on one set of generated data, returns a dictionary of information about the run."""

    alpha = 0.05
    p_boot_thr = 0.05
    #generate data, now with the option to accept external data
    if data is None:
        (ind_data,dep_data,sn_stats) = gen_data(nsubj,nobs,nfeat,slope,signal,
                                            mnoise=mnoise,contvar=contvar,
                                            item_inds=item_inds,mod_cont=mod_cont,I=I,S=S)
    else:
        (ind_data,dep_data,sn_stats) = data

    #set up formulas
    #This structure models subject level noise on the slope and intercept
    re_formula = '(0+beh|subj)'
    if mod_cont == True:
        fe = 'beh*cont'
        re_formula += ' + (0+cont|subj) + (0+beh:cont|subj)'
    else:
        fe = 'beh'
    
    fe_formula = 'val ~ %s'%fe
    if mod_mnoise == False:
        re_formula='(1|subj)'
    else:
        #re_formula +=' + (1|subj)'
        re_formula = '(beh|subj)'
    
    #only intercept effects for item because items aren't repeated
    #see http://talklab.psy.gla.ac.uk/KeepItMaximalR2.pdf page 13
    if item_inds == True:
        re_formula += ' + (1|item)'
        
    print fe_formula
    print re_formula
    ##LMER part
    #create lmer object
    me_s = meld.MELD(fe_formula, re_formula, 'subj',
                    dep_data, ind_data, factors = {'subj':None},
                    use_ranks=use_ranks,
                    feat_nboot=feat_nboot,feat_thresh=feat_thresh,
                    do_tfce=do_tfce,
                    connectivity=connectivity,shape=None,
                    dt=dt,E=E, H=H,
                    n_jobs=n_jobs)
    
    #run LMER
    me_s.run_perms(nperms)
    me_ps = me_s.pvals
    me_pfs = me_s.p_features

    #evaluate LMER results
    under = {}
    tprs = {}
    fprs = {}
    me_thr = {}
    mcc = {}
    me_et = {}
    peak_mcc={}
    peak_mcc_thr ={}
    #get error terms
    for fac in me_s.terms:
        
        if fac != '(Intercept)':
            
            statmap = 1-me_pfs[fac]
            under[fac],tprs[fac],fprs[fac],me_thr[fac],mcc[fac]=get_ROC(statmap, signal*slope,num=1000,ret_thr = True,ret_mcc=True)
            me_thr[fac]= 1-me_thr[fac]
            me_et[fac]=get_error_terms((statmap)>=(1.-p_boot_thr),np.abs(signal*slope)>0)
            peak_mcc[fac] = np.max(mcc[fac])
            peak_mcc_thr[fac] = me_thr[fac][np.argmax(mcc[fac])]
        else:
            under[fac] = np.nan
            tprs[fac] = np.nan
            fprs[fac] = np.nan

    ##ttest part
    print "Running across subject ttest on within subject, within condition means"
    verbose = 10
    
    #get the perms
    perms = gen_perms(np.asarray(ind_data), 'subj', ntperms)
    
    print "T-testing the permuted values"
    tvals = Parallel(n_jobs=n_jobs,verbose=10)(delayed(eval_perm)(perm,ind_data,dep_data)
                                       for perm in perms)
    tvals = np.concatenate(tvals)
    
    if do_ttfce==True:
        con = cl.sparse_dim_connectivity([cl.simple_neighbors_1d(n) for n in tvals[0,:,:].shape])
        print "TFCEing the T-value maps"
        tclust = Parallel(n_jobs=n_jobs,verbose=10)(delayed(run_tfce)(t,con,dt,E,H)
                                           for t in tvals)
        tclust = np.concatenate(tclust)
    
        #bulky, but unfortunately, I need to save out the terms for ttest with and without tfce
    
        thr_min_tfce = stats.scoreatpercentile(np.abs(tclust).reshape((len(tclust),-1)).max(1), 0)
        thr_alpha_tfce = stats.scoreatpercentile(np.abs(tclust).reshape((len(tclust),-1)).max(1), ((1-(alpha)/2.)*100))
        #thr_list = np.linspace(0,thr_min,100)
        #alpha_list = np.zeros(len(thr_list))
        thr_list_tfce = [stats.scoreatpercentile(np.abs(tclust).reshape((len(tclust),-1)).max(1), ((1-(i)/2.)*100))
                             for i in np.linspace(0,1,901)]
        thr_list_tfce.extend(list(np.linspace(thr_min_tfce,0,100)))
        alpha_list_tfce = np.append(np.linspace(0,1,901),np.ones(100))
        tt_under_tfce,tt_tprs_tfce,tt_fprs_tfce,tt_mcc_tfce=get_ROC(np.abs(tclust[0]),signal*slope,thr=thr_list_tfce,ret_mcc = True)
        tt_et_tfce=get_error_terms(np.abs(tclust[0])>=thr_alpha_tfce,np.abs(signal*slope)>0)
    
    thr_min_ = stats.scoreatpercentile(np.abs(tvals).reshape((len(tvals),-1)).max(1), 0)
    thr_alpha = stats.scoreatpercentile(np.abs(tvals).reshape((len(tvals),-1)).max(1), ((1-(alpha)/2.)*100))
    #thr_list = np.linspace(0,thr_min,100)
    #alpha_list = np.zeros(len(thr_list))
    thr_list = [stats.scoreatpercentile(np.abs(tvals).reshape((len(tvals),-1)).max(1), ((1-(i)/2.)*100))
                         for i in np.linspace(0,1,901)]
    thr_list.extend(list(np.linspace(tvals,0,100)))
    alpha_list = np.append(np.linspace(0,1,901),np.ones(100))
    tt_under,tt_tprs,tt_fprs,tt_mcc=get_ROC(np.abs(tvals[0]),signal*slope,thr=thr_list,ret_mcc = True)
    tt_et=get_error_terms(np.abs(tvals[0])>=thr_alpha,np.abs(signal*slope)>0)
    
    res = {'slope':slope,
           'signal_shape':signal.shape,
           'nsubj':nsubj,
           'nobs':nobs,
           'mnoise':mnoise,
           'inoise':I,
           'snoise':S,
           'contvar':contvar,
           'mod_mnoise':mod_mnoise,
           'mod_cont':mod_cont,
           'ranks':use_ranks,
           'sn_stats':sn_stats,
           'model':me_s._formula_str,
           'glm_model':fe_formula,
           'nperms':nperms,
           'nboots':nboots,
           'alpha':alpha,
           'p_boot_thr':p_boot_thr,
           'meld_et':me_et,
           'meld_pvals':me_ps,
         #  'lfas_et':lfas_et,
           'tt_et':tt_et,
           'meld_peak_mcc':peak_mcc,
           'meld_peak_mcc_thr':peak_mcc_thr
           }

    res['meld_auc']=under
    #res['lfas_auc']=lfas_under
    res['tt_auc']=tt_under
    if do_ttfce == True:
        res['tt_et_tfce']=tt_et_tfce
        res['tt_auc_tfce']=tt_under_tfce
    if save_ROC == True:
        res['meld_tprs']=tprs
        res['meld_fprs']=fprs
        res['meld_thr']=me_thr
        res['meld_mcc']=mcc
       # res['lfas_tprs']=lfas_tprs
       # res['lfas_fprs']=lfas_fprs
        res['tt_tprs']=tt_tprs
        res['tt_fprs']=tt_fprs
        res['tt_thr_list']=thr_list
        res['tt_alpha_list']=alpha_list
        res['tt_mcc']=tt_mcc
        if do_ttfce==True:
            res['tt_tprs_tfce']=tt_tprs_tfce
            res['tt_fprs_tfce']=tt_fprs_tfce
            res['tt_thr_list_tfce']=thr_list_tfce
            res['tt_alpha_list_tfce']=alpha_list_tfce
            res['tt_mcc_tfce']=tt_mcc_tfce
        
        
    if save_lfud== True:
        res['meld']=me_s
    if save_maps == True:
        res['meld_brmap']=me_s.boot_ratio
        #res['meld_fdrmap']=me_s.fdr_boot
        res['meld_pmap']=me_pfs
        #res['meld_qmap']=meld_qvals
        #res['lfas_map']=lfas_map
        #res['_tmap']=ttestres
        res['tt_map']=tvals[0,:,:]
        res['tt_alpha']=thr_alpha
        # get the feature ts for each boot, we'll figure out what to do with them
        tfs = me_s._tb[0].__array_wrap__(np.hstack(me_s._tb))
        names = [n for n in tfs.dtype.names if n != '(Intercept)']
        tfsrs = []
        for i,n in enumerate(names):
            fmask = me_s._fmask[i]
            tf = tfs[n].reshape((len(me_s._tb),-1))[:,fmask]
            tfsrs.append(tf)
        tfsrs = np.rec.fromarrays(tf, names=','.join(names))

        res['meld_tfs']=tfsrs
        if do_ttfce==True:
            res['tt_map_tfce']=tclust[0,:,:]
            res['tt_alpha_tfce']=thr_alpha_tfce

    return res 


# In[19]:

if __name__=='__main__':
    outname = 'sim_results_%d_%d_%d_%d_%d_%d_%s_%0.1f_tfre'
    signal_name = 'cfrac'
    prop = 0.001
    slopes = np.array([0.1])
    do_ttfce = True
    mod_mnoise = True
    n_jobs = -1
    nruns = 5 
    nperms = 500
    nboots = 0
    ntperms = 500
    contvar = False
    item_inds = True
    mnoise = True
    I = 1.0
    S = 0.1


    nsubjs = np.array([9])
    nobses = np.array([50])
    use_ranks = False
    nfeat = (100,100)
    con = cl.sparse_dim_connectivity([cl.simple_neighbors_1d(n) for n in nfeat])
    


    i = 0
    j = 0
        
    try:
        results
    except NameError:
        results = []
        
    save_lfud = False
    save_maps = True
    write_each = True
    save_ROC = True
    do_tfce = True

    feat_thresh = 0.05


    total_runs = len(slopes)*len(nsubjs)*len(nobses)*nruns
    start_time = time.time()
    ltime = time.localtime(start_time)
    outname = outname%(ltime.tm_year,ltime.tm_mon,ltime.tm_mday,ltime.tm_hour,ltime.tm_min,ltime.tm_sec,signal_name,slopes[0])
    z = 0
    #make it harder to overwrite the default outputname
    while os.path.exists(outname+'.pickle'):
        z +=1
        outname = 'sim_results_%d_%d_%d_%d_%d_%d_%d'%(ltime.
    tm_year,ltime.tm_mon,ltime.tm_mday,
	                                              ltime.tm_hour,ltime.tm_min,ltime.tm_sec,z)
    outname += '.pickle'


    sys.stdout.write("Starting %3d runs\n"%(total_runs))
    sys.stdout.flush()

    if contvar == True:
        mod_cont=True
    else:
        mod_cont=False

    #if mnoise == True:
    #    mod_mnoise=True
    #else:
    #    mod_mnoise=False

    #for signal in signals:

    for nobs in nobses:
        for slope in slopes:
	    for nsubj in nsubjs:
	        for run in range(nruns):
	            signal = hide_blobs(cfrac=np.multiply(prop,100))
	            run_time = time.time()
	            res = test_sim_dat(nsubj,nobs,slope,signal,mnoise=mnoise,
	                               contvar=contvar,item_inds=item_inds,
	                               mod_mnoise=mod_mnoise,mod_cont=mod_cont,
	                               use_ranks=use_ranks,nfeat=nfeat,save_lfud=save_lfud,
	                               n_jobs=n_jobs,save_maps=save_maps,nperms=nperms,nboots=nboots,ntperms=ntperms,
	                               save_ROC=save_ROC,I=I,S=S,do_tfce=do_tfce, do_ttfce=do_ttfce,feat_thresh=feat_thresh,connectivity=con,
	                               data=None)
	            res['signal_name'] = signal_name
	            res['prop'] = prop
	            res['run']=j
	            res['time']=(time.time()-run_time)
	            results.append(res)
	            if ((write_each == True) & (save_lfud == False)):
	                with open(outname,'wb') as handle:
	                    pickle.dump(results, handle, protocol=2)
	            i += 1
	            sys.stdout.write('Run %d finished in %.2f sec\n'%(i,time.time()-run_time))
	            sys.stdout.write('%.2f sec elapsed since start\n'%(time.time()-start_time))
	            sys.stdout.flush()

	            print "Completed %3d runs of %3d total"%(i,total_runs)                                        
	        j += 1

	                            
    #print "AUC Preview"
    #print "MELD\tGLM\tGLM"
    #for i in range(len(results)):
    #    print "%0.2f\t%0.2f\t%0.2f"%(results[i]['meld_auc'],results[i]['glm_auc'],results[i]['glm_fdr_auc'])
    if not write_each:
        if not save_lfud:
	    with open(outname,'wb') as handle:
	        pickle.dump(results, handle, protocol=2)


    # In[ ]:



