# let's be brutally honest here.  I wrote all this before I understood sklearn Pipelines.
# if I had understood I could have saved myself a lot of trouble.
# a really great student short project would be to replace the gene subsetting system with a custom
# transformation and then just stick it in a Pipeline with whatever you want
# JGF July 2021

from __future__ import division

import pandas as pd
import re
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_absolute_error as score_MAE
from sklearn.metrics import mean_squared_error as score_MSE 
from sklearn.metrics import median_absolute_error as score_MED
from sklearn.metrics import r2_score as score_R2
from sklearn.metrics import accuracy_score as score_ACC
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import *
import scipy.stats as stats
from datetime import datetime
import os.path

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,RepeatedKFold,LeaveOneOut,LeaveOneGroupOut

# this is useful because if you do anything other than LeaveOneOut the output of ensemble.predict is a list of lists
flatten = lambda l: pd.Series([item for sublist in l for item in sublist]) 

# this version conforms to the sklearn fit / predict interface instead of just pretending to like before;
# now you can use .fit() and it does something, and .predict() no longer applies a fit for you
# it also implements rank based transformation of the data, or quantile based transformations,
# but doesn't allow you to do both at once
# These transformations might hopefully give you a bit more freedom from batch effects
# Finally this version also implements FPKM to TPM conversion should you desire
 
class subset_genes_ensemble(BaseEstimator, ClassifierMixin):
    
    def __init__(self, clf=None, class_size=20, subset_min=0, subset_fold=0, dataxform_log=False, dataxform_fpkmToTpm=False, dataxform_rank=False, dataxform_quantile=False, verbose=False, seed=42):
        assert ( int(dataxform_rank) + int(dataxform_quantile) < 2), "We can only do one of rank and quantile normalization, not both" 
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.dataxform_log = dataxform_log
        self.dataxform_fpkmToTpm = dataxform_fpkmToTpm
        self.dataxform_rank = dataxform_rank
        self.dataxform_quantile = dataxform_quantile
        self.clf = clf
        self.class_size = class_size
        self.verbose = verbose
        self.seed = seed
        
    def _fpkmToTpm(self, fpkm):
        # takes ndarray, assumes rows are samples, columns are genes
        # turns fpkm into tpm - following https://haroldpimentel.wordpress.com/2014/05/08/what-the-fpkm-a-review-rna-seq-expression-units/
        return np.exp( np.log(fpkm) - np.log(fpkm.sum(axis=1).reshape(-1,1).repeat(fpkm.shape[1],axis=1)) + np.log(1e6))
   
    
    def _expr_levels(self, data):
        # note the order of the if statements implies the order of execution for combinations of transforms
        # only rank and quantile are not allowed simultaneously via assertion in __init__
        d=data.copy()
        if self.dataxform_fpkmToTpm:
            d = self._fpkmToTpm(d)
        if self.dataxform_rank:
            d = d.rank(axis=1)
        if self.dataxform_log:
            d = np.log2(d + 1.)
        if self.dataxform_quantile:
            check_is_fitted(self, 'qtrans_')
            if self.qtrans_.quantiles_.shape[1] == d.shape[1]:
                # if the xformer was fit on the same size data, asssume it matches up
                d = self.qtrans_.transform(d)
                d = pd.DataFrame( d, index=data.index, columns=data.columns)
            else:
                # do it gene by gene where the genes match up between this data and training data
                tmp = {}
                keep = np.copy(self.qtrans_.quantiles_)
                qt = self.qtrans_
                for agene in np.intersect1d( self.qdict_.keys(), d.columns):
                    qt.quantiles_ = self.qdict_[agene]
                    tmp[agene] = flatten(  qt.transform( d.loc[:,agene].values.reshape(-1, 1) )  )
                    #reshape because this is a single feature and sklearn expects this
                    # but then flatten because otherwise dataframe creation barfs
                d = pd.DataFrame( tmp )
                d.index = data.index
                qt.quantiles_ = keep # return qt to its original state
                    
        
        return d
        
    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame 
    def _subset_genes(self, data, verbose=False, these_genes=np.array(False) ):
        
        calc_subset = True
        if these_genes.any(): # just get these_genes, without calculating
            calc_subset = False
        else: # calculate the these_genes list based on the other parameters    
            genes = data  # operate selection criteria on non-transformed FPKM
            eps = 0.001
            has_start_gt = genes > self.subset_min
            mvt = genes.min()
            mvt[ mvt<eps ] = eps
            has_fold_change = (genes.max() / mvt)  > self.subset_fold            
            aset = (has_start_gt.any(axis=0) & has_fold_change)
            these_genes = aset[ aset ].index # get list of gene names that meet criteria
            
        genes = self._expr_levels(data) # do the final op on user-selected xform
            
        subgenes = genes.loc[:,genes.columns.intersection(these_genes)]

        if verbose & calc_subset:
            print('using {} genes in subset requiring a max FPKM > {} and > {}-fold change between max and min samples '.format(len(these_genes), self.subset_min, self.subset_fold))

        if (not calc_subset) & (subgenes.shape[1] < len(these_genes)):
            print('warning: only {} of {} requested genes present in data during subsetting; filling with 0.0'.format(subgenes.shape[1],len(these_genes)))
            missing = np.setdiff1d( these_genes, subgenes.columns)
            if verbose: 
                print(missing)
            for amiss in missing:
                subgenes[amiss] = 0.0

        subgenes =  subgenes.sort_index(axis=1)  # need to enforce constant variable order
        # especially now that we allow missing variables in subset_genes
        
        return subgenes
    
    def _get_bounds(self, minimum,maximum,offset):

        interval = self.class_size
        done = False
        lower = minimum
        while (not done):
            if (lower==minimum):
                if (offset>0):
                    upper = lower+offset
                else:
                    upper = lower+interval

                bounds = np.array([[lower,upper-1]])
                lower = upper
                continue
            else:
                upper = min(lower+interval,maximum+1)

            done = upper>maximum
            bounds = np.append(bounds,[[lower,upper-1]],axis=0)

            lower = upper
        return bounds;

    def _trim_bytes(self):
        # make the ensemble smaller for binary dump by deleting things needed only for training and analysis
        # keep all the stuff thats needed for predicting with a trained classifier -- about 3.5 - 4GB for an LDA ensemble
        for a_clf in self.classifiers_:
            del a_clf.covariance_
    
        del self.train_data_
        del self.train_label_

    def _trim_classifiers(self):
        # make the ensemeble TRULY small for binary dump by deleting the classifiers themselves
        # but by keeping all the training data and settings the classifier can be reinstantiated with refit()
        del self.classifiers_
        
    def refit(self):
        check_is_fitted(self, 'train_data_')
        check_is_fitted(self, 'train_label_')
        self.fit( self.train_data_, self.train_label_, verbose=False, these_genes=self.genecolumns_ )
        
    def predict(self, X):
        check_is_fitted(self, 'genecolumns_')
        check_is_fitted(self, 'classifiers_')
        check_is_fitted(self, 'bounds_')
        check_is_fitted(self, 'class_names_')
        check_is_fitted(self, 'label_encoder_')
        
        X_sub = self._subset_genes(X, verbose=True, these_genes=self.genecolumns_)

        votes = []
        for _ in range(X_sub.shape[0]): # make a votes list of lists as long as the number of test samples
            votes.append([])
            
        # For each partitioning of the output space, predict with that member of the ensemble
        for offset in range(0,self.class_size):  
            
            a_clf = self.classifiers_[offset]
            bounds = self.bounds_[offset]
            class_names = self.class_names_[offset]
            
            predictions = self.label_encoder_[offset].inverse_transform( 
                a_clf.predict(X_sub) )

            """        
            Each vote for age i is stored as an single instance of integer i 
            (i.e. by analogy, a slip with the age is placed in a ballot box) 
            so that the mean and median predicted age can be taken in addition to 
            or instead of the mode.
            """
            for k, class_predict in enumerate(predictions):
                
                # Generate constituent integer ages from the predicted age class
                bnd_predict = bounds[class_names == class_predict][0]
                this_vote = np.arange(bnd_predict[0],bnd_predict[1]+1,1)

                # Store predicted integer ages
                votes[k] = np.append(votes[k],this_vote)

        # Take the mode (or median or mean or some more elaborate voting scheme if you like) integer predicted age
        # as the final predicted age
        age_predict = [ stats.mode(vs).mode[0] for vs in votes]
        self.votes_ = votes
        
        return age_predict
    
    def fit(self, X, y, verbose=False, these_genes=np.array(False)):
        check_is_fitted(self, 'subset_min')
        check_is_fitted(self, 'subset_fold')
        check_is_fitted(self, 'dataxform_log')
        check_is_fitted(self, 'dataxform_fpkmToTpm')
        check_is_fitted(self, 'dataxform_rank')
        check_is_fitted(self, 'dataxform_quantile')

        np.random.seed(self.seed) # this is needed for reproducability of results for classifiers that use a random number generator
        # this seed can be modified during ensemeble initialization, if no argument is set seed defaults to The Answer to Life, The Universe, and Everything 

        if self.dataxform_quantile:
            self.qtrans_ = QuantileTransformer().fit(X)
            self.qdict_ = {} #keeping this dictionary will allow us to fit gene by gene later
            # which will be useful for cases when we try to predict with a different dataset than the fitting set
            for k, agene in enumerate(X.columns):
                self.qdict_[agene] = self.qtrans_.quantiles_[:,k].reshape(-1,1) #reshape because this is a single feature and sklearn expects this
            
        if these_genes.any(): # use the given subset
            X_sub = self._subset_genes(X, verbose=self.verbose, these_genes=these_genes)
        else: # train the gene subset on this data too
            X_sub = self._subset_genes(X, verbose=self.verbose)
        
        self.genecolumns_ = X_sub.columns
        self.train_data_ = X_sub
        self.train_label_ = y
        self.classifiers_ = []
        self.bounds_ = []
        self.class_names_ = []
        self.label_encoder_ = []
        
        n_samp = len(y)
        age_min = min(y)
        age_max = max(y)

        # For each partitioning of the output space, create a member of the ensemble
        for offset in range(0,self.class_size):  
            
            if verbose:
                print("ensemble member #{}".format(offset))
                
            a_clf = clone(self.clf)
            
            bounds = self._get_bounds(age_min,age_max,offset)
            n_classes = len(bounds)
            train_class = np.empty(n_samp,dtype=object)
            class_names = np.empty(n_classes,dtype=object)

            for i in range(0,n_classes):
                lower = bounds[i][0];
                upper = bounds[i][1];
                class_names[i] = str(lower) + '-' + str(upper)
                in_class = np.logical_and(y>=lower,y<=upper)
                train_class[in_class] = class_names[i]
            
            lenc = LabelEncoder().fit(train_class)
            targets = lenc.transform(train_class)

            a_clf.fit(X_sub,targets)
            self.classifiers_.append(a_clf)
            self.bounds_.append(bounds)
            self.class_names_.append(class_names)
            self.label_encoder_.append(lenc)
            
        return self



class subset_genes_LinRegr(LinearRegression):
    
    def __init__(self, subset_min=0, subset_fold=0, dataxform_log=False, dataxform_fpkmToTpm=False, dataxform_rank=False, dataxform_quantile=False, verbose=False, seed=42, fit_intercept=True, normalize=False, copy_X=True, n_jobs=1):
        assert ( int(dataxform_rank) + int(dataxform_quantile) < 2), "We can only do one of rank and quantile normalization, not both" 
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.dataxform_log = dataxform_log
        self.dataxform_fpkmToTpm = dataxform_fpkmToTpm
        self.dataxform_rank = dataxform_rank
        self.dataxform_quantile = dataxform_quantile
        self.verbose = verbose
        self.seed = seed
        super(subset_genes_LinRegr, self).__init__(
                    fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs )
        
    def _fpkmToTpm(self, fpkm):
        # takes ndarray, assumes rows are samples, columns are genes
        # turns fpkm into tpm - following https://haroldpimentel.wordpress.com/2014/05/08/what-the-fpkm-a-review-rna-seq-expression-units/
        return np.exp( np.log(fpkm) - np.log(fpkm.sum(axis=1).reshape(-1,1).repeat(fpkm.shape[1],axis=1)) + np.log(1e6))
   
    
    def _expr_levels(self, data):
        # note the order of the if statements implies the order of execution for combinations of transforms
        # only rank and quantile are not allowed simultaneously via assertion in __init__
        d=data.copy()
        if self.dataxform_fpkmToTpm:
            d = self._fpkmToTpm(d)
        if self.dataxform_rank:
            d = d.rank(axis=1)
        if self.dataxform_log:
            d = np.log2(d + 1.)
        if self.dataxform_quantile:
            check_is_fitted(self, 'qtrans_')
            if self.qtrans_.quantiles_.shape[1] == d.shape[1]:
                # if the xformer was fit on the same size data, asssume it matches up
                d = self.qtrans_.transform(d)
                d = pd.DataFrame( d, index=data.index, columns=data.columns)
            else:
                # do it gene by gene where the genes match up between this data and training data
                tmp = {}
                keep = np.copy(self.qtrans_.quantiles_)
                qt = self.qtrans_
                for agene in np.intersect1d( self.qdict_.keys(), d.columns):
                    qt.quantiles_ = self.qdict_[agene]
                    tmp[agene] = flatten(  qt.transform( d.loc[:,agene].values.reshape(-1, 1) )  )
                    #reshape because this is a single feature and sklearn expects this
                    # but then flatten because otherwise dataframe creation barfs
                d = pd.DataFrame( tmp )
                d.index = data.index
                qt.quantiles_ = keep # return qt to its original state
                    
        
        return d
        
    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame 
    def _subset_genes(self, data, verbose=False, these_genes=np.array(False) ):
        
        calc_subset = True
        if these_genes.any(): # just get these_genes, without calculating
            calc_subset = False
        else: # calculate the these_genes list based on the other parameters    
            genes = data  # operate selection criteria on non-transformed FPKM
            eps = 0.001
            has_start_gt = genes > self.subset_min
            mvt = genes.min()
            mvt[ mvt<eps ] = eps
            has_fold_change = (genes.max() / mvt)  > self.subset_fold            
            aset = (has_start_gt.any(axis=0) & has_fold_change)
            these_genes = aset[ aset ].index # get list of gene names that meet criteria
            
        genes = self._expr_levels(data) # do the final op on user-selected xform
            
        subgenes = genes.loc[:,genes.columns.intersection(these_genes)]

        if verbose & calc_subset:
            print('using {} genes in subset requiring a max FPKM > {} and > {}-fold change between max and min samples '.format(len(these_genes), self.subset_min, self.subset_fold))

        if (not calc_subset) & (subgenes.shape[1] < len(these_genes)):
            print('warning: only {} of {} requested genes present in data during subsetting; filling with 0.0'.format(subgenes.shape[1],len(these_genes)))
            missing = np.setdiff1d( these_genes, subgenes.columns)
            if verbose: 
                print(missing)
            for amiss in missing:
                subgenes[amiss] = 0.0

        subgenes =  subgenes.sort_index(axis=1)  # need to enforce constant variable order
        # especially now that we allow missing variables in subset_genes
        
        return subgenes

        

    def predict(self, X):
        """Perform regression on samples in X.
        For an one-class model, +1 (inlier) or -1 (outlier) is returned.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).
        Returns
        -------
        y_pred : array, shape (n_samples,)
        """
        check_is_fitted(self, 'genecolumns_')
        X_sub = self._subset_genes(X, these_genes=self.genecolumns_)
        #X_sub = super(subset_genes_LinRegr, self)._validate_for_predict(X_sub)
        return super(subset_genes_LinRegr, self).predict(X_sub)
    
    def fit(self, X, y, verbose=False, these_genes=np.array(False)):
        check_is_fitted(self, 'subset_min')
        check_is_fitted(self, 'subset_fold')
        check_is_fitted(self, 'dataxform_log')
        check_is_fitted(self, 'dataxform_fpkmToTpm')
        check_is_fitted(self, 'dataxform_rank')
        check_is_fitted(self, 'dataxform_quantile')

        np.random.seed(self.seed) # this is needed for reproducability of results for classifiers that use a random number generator
        # this seed can be modified during ensemeble initialization, if no argument is set seed defaults to The Answer to Life, The Universe, and Everything 

        if self.dataxform_quantile:
            self.qtrans_ = QuantileTransformer().fit(X)
            self.qdict_ = {} #keeping this dictionary will allow us to fit gene by gene later
            # which will be useful for cases when we try to predict with a different dataset than the fitting set
            for k, agene in enumerate(X.columns):
                self.qdict_[agene] = self.qtrans_.quantiles_[:,k].reshape(-1,1) #reshape because this is a single feature and sklearn expects this
            
        if these_genes.any(): # use the given subset
            X_sub = self._subset_genes(X, verbose=self.verbose, these_genes=these_genes)
        else: # train the gene subset on this data too
            X_sub = self._subset_genes(X, verbose=self.verbose)
        
        self.genecolumns_ = X_sub.columns
        self.train_data_ = X_sub
        self.train_label_ = y

        super(subset_genes_LinRegr, self).fit(X_sub, y)

        return self

    def predict(self, X):
        check_is_fitted(self, 'genecolumns_')
        
        X_sub = self._subset_genes(X, verbose=True, these_genes=self.genecolumns_)

        return super(subset_genes_LinRegr, self).predict(X_sub)



class subset_genes_ElasticNet(ElasticNet):
    
    def __init__(self, subset_min=0, subset_fold=0, dataxform_log=False, dataxform_fpkmToTpm=False, dataxform_rank=False, dataxform_quantile=False, verbose=False, seed=42,  alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=1e-4, warm_start=False, positive=False,selection='cyclic'):
        assert ( int(dataxform_rank) + int(dataxform_quantile) < 2), "We can only do one of rank and quantile normalization, not both" 
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.dataxform_log = dataxform_log
        self.dataxform_fpkmToTpm = dataxform_fpkmToTpm
        self.dataxform_rank = dataxform_rank
        self.dataxform_quantile = dataxform_quantile
        self.verbose = verbose
        self.seed = seed
        super(subset_genes_ElasticNet, self).__init__(
                 alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept,
                 normalize=normalize, precompute=precompute, max_iter=max_iter,
                 copy_X=copy_X, tol=tol, warm_start=warm_start, positive=positive,
                 random_state=seed, selection=selection )
        
    def _fpkmToTpm(self, fpkm):
        # takes ndarray, assumes rows are samples, columns are genes
        # turns fpkm into tpm - following https://haroldpimentel.wordpress.com/2014/05/08/what-the-fpkm-a-review-rna-seq-expression-units/
        return np.exp( np.log(fpkm) - np.log(fpkm.sum(axis=1).reshape(-1,1).repeat(fpkm.shape[1],axis=1)) + np.log(1e6))
   
    
    def _expr_levels(self, data):
        # note the order of the if statements implies the order of execution for combinations of transforms
        # only rank and quantile are not allowed simultaneously via assertion in __init__
        d=data.copy()
        if self.dataxform_fpkmToTpm:
            d = self._fpkmToTpm(d)
        if self.dataxform_rank:
            d = d.rank(axis=1)
        if self.dataxform_log:
            d = np.log2(d + 1.)
        if self.dataxform_quantile:
            check_is_fitted(self, 'qtrans_')
            if self.qtrans_.quantiles_.shape[1] == d.shape[1]:
                # if the xformer was fit on the same size data, asssume it matches up
                d = self.qtrans_.transform(d)
                d = pd.DataFrame( d, index=data.index, columns=data.columns)
            else:
                # do it gene by gene where the genes match up between this data and training data
                tmp = {}
                keep = np.copy(self.qtrans_.quantiles_)
                qt = self.qtrans_
                for agene in np.intersect1d( self.qdict_.keys(), d.columns):
                    qt.quantiles_ = self.qdict_[agene]
                    tmp[agene] = flatten(  qt.transform( d.loc[:,agene].values.reshape(-1, 1) )  )
                    #reshape because this is a single feature and sklearn expects this
                    # but then flatten because otherwise dataframe creation barfs
                d = pd.DataFrame( tmp )
                d.index = data.index
                qt.quantiles_ = keep # return qt to its original state
                    
        
        return d
        
    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame 
    def _subset_genes(self, data, verbose=False, these_genes=np.array(False) ):
        
        calc_subset = True
        if these_genes.any(): # just get these_genes, without calculating
            calc_subset = False
        else: # calculate the these_genes list based on the other parameters    
            genes = data  # operate selection criteria on non-transformed FPKM
            eps = 0.001
            has_start_gt = genes > self.subset_min
            mvt = genes.min()
            mvt[ mvt<eps ] = eps
            has_fold_change = (genes.max() / mvt)  > self.subset_fold            
            aset = (has_start_gt.any(axis=0) & has_fold_change)
            these_genes = aset[ aset ].index # get list of gene names that meet criteria
            
        genes = self._expr_levels(data) # do the final op on user-selected xform
            
        subgenes = genes.loc[:,genes.columns.intersection(these_genes)]

        if verbose & calc_subset:
            print('using {} genes in subset requiring a max FPKM > {} and > {}-fold change between max and min samples '.format(len(these_genes), self.subset_min, self.subset_fold))

        if (not calc_subset) & (subgenes.shape[1] < len(these_genes)):
            print('warning: only {} of {} requested genes present in data during subsetting; filling with 0.0'.format(subgenes.shape[1],len(these_genes)))
            missing = np.setdiff1d( these_genes, subgenes.columns)
            if verbose: 
                print(missing)
            for amiss in missing:
                subgenes[amiss] = 0.0

        subgenes =  subgenes.sort_index(axis=1)  # need to enforce constant variable order
        # especially now that we allow missing variables in subset_genes
        
        return subgenes

        

    def predict(self, X):
        """Perform regression on samples in X.
        For an one-class model, +1 (inlier) or -1 (outlier) is returned.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).
        Returns
        -------
        y_pred : array, shape (n_samples,)
        """
        check_is_fitted(self, 'genecolumns_')
        X_sub = self._subset_genes(X, these_genes=self.genecolumns_)
        #X_sub = super(subset_genes_LinRegr, self)._validate_for_predict(X_sub)
        return super(subset_genes_ElasticNet, self).predict(X_sub)
    
    def fit(self, X, y, verbose=False, these_genes=np.array(False)):
        check_is_fitted(self, 'subset_min')
        check_is_fitted(self, 'subset_fold')
        check_is_fitted(self, 'dataxform_log')
        check_is_fitted(self, 'dataxform_fpkmToTpm')
        check_is_fitted(self, 'dataxform_rank')
        check_is_fitted(self, 'dataxform_quantile')

        np.random.seed(self.seed) # this is needed for reproducability of results for classifiers that use a random number generator
        # this seed can be modified during ensemeble initialization, if no argument is set seed defaults to The Answer to Life, The Universe, and Everything 

        if self.dataxform_quantile:
            self.qtrans_ = QuantileTransformer().fit(X)
            self.qdict_ = {} #keeping this dictionary will allow us to fit gene by gene later
            # which will be useful for cases when we try to predict with a different dataset than the fitting set
            for k, agene in enumerate(X.columns):
                self.qdict_[agene] = self.qtrans_.quantiles_[:,k].reshape(-1,1) #reshape because this is a single feature and sklearn expects this
            
        if these_genes.any(): # use the given subset
            X_sub = self._subset_genes(X, verbose=self.verbose, these_genes=these_genes)
        else: # train the gene subset on this data too
            X_sub = self._subset_genes(X, verbose=self.verbose)
        
        self.genecolumns_ = X_sub.columns
        self.train_data_ = X_sub
        self.train_label_ = y

        super(subset_genes_ElasticNet, self).fit(X_sub, y)

        return self

    def predict(self, X):
        check_is_fitted(self, 'genecolumns_')
        
        X_sub = self._subset_genes(X, verbose=True, these_genes=self.genecolumns_)

        return super(subset_genes_ElasticNet, self).predict(X_sub)
