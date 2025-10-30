import json
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from smt.surrogate_models import KPLS, KPLSK, KRG, MixIntKernelType, MixHrcKernelType
from scipy.optimize import minimize
from .mixturev2 import MOE
from smt.utils.design_space import (
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
    CategoricalVariable,
)
from smt.applications.mixed_integer import (
    MixedIntegerKrigingModel,
    MixedIntegerSamplingMethod,
)

class GPops_dev(): 

    def __init__(
        self,
        training_data: Dict[int, Dict[str, Union[List, np.ndarray]]],
        bandwidth: float = 0.1,
        bandwidtha: float = 0.1,
        bandwidths: float = 0.1,
        variance_mode: str = 'target',
        normalization: str = 'mean/var',
        weight_dilution_strategy: Union[int, str] = 'None',
        number_of_function_evaluations: float = 50,
        n_clusters: int = 1,
        weighting_method: str = "None", 
        surrogate_model_type: str = 'KPLS',
        gp_scaling: bool = True,
        ignore_target_model: bool = False, #variable added in cases where target model is assumed to be inaccurate due to minimal number of I/O pairings
        alpha_t: float = 1,
        alpha_a: float = 0,
        alpha_s: float = 0,
        min_accuracy: float = 0.5,
        max_vars : float = 2,
        smooth_recombination: bool =  False,
        smooth_weights: bool = True,
        ens_kernel: str= 'EPAN',
        **kwargs
    ):
        """
        Transfer learning in surrogate modeling from "Transfer Learning in Surrogate Modeling with
        Emphasis on Aircraft Design" by Tfaily et al. 
        https://www.gerad.ca/en/papers/G-2025-50

        """

        if kwargs.get('instance_features') is not None:
            raise NotImplementedError()
        super().__init__(**kwargs)
        self.training_data = training_data
        self.gp_scaling = gp_scaling
        self.alpha_t = alpha_t
        self.alpha_a = alpha_a
        self.alpha_s = alpha_s
        alpha_sum = self.alpha_t +self.alpha_a+self.alpha_s
        self.alpha_t = self.alpha_t/alpha_sum
        self.alpha_a = self.alpha_a/alpha_sum
        self.alpha_s = self.alpha_s/alpha_sum
        self.min_accuracy = min_accuracy
        self.max_vars = max_vars
        self.smooth_recombination = smooth_recombination
        self.smooth_weights = smooth_weights
        self.ens_kernel = ens_kernel # default ensemble kernel is the epanechnikov kernel

        self.bandwidth = bandwidth #bandwidth for shape (discordant pairs)
        self.bandwidtha = bandwidtha # bandwidth for accuracy
        self.bandwidths = bandwidths # bandwidth for vairance
        self.rng = np.random.RandomState(1)#self.seed
        self.variance_mode = variance_mode
        self.normalization = normalization
        self.weight_dilution_strategy = weight_dilution_strategy
        self.number_of_function_evaluations = number_of_function_evaluations
        self.ignore_target_model = ignore_target_model
        self.weighting_method = weighting_method # options:DISC-PRS E-REL: relative error
        if self.weighting_method not in ['None', 'DISC-PRS', 'E-REL']:
            print('WARNING: incorrect weighting method. Assigning discourdant pairs as the weighting method. ')
            self.weighting_method = 'None'
        self.n_clusters = n_clusters
        self.clustering = False
        if n_clusters > 1:
            self.clustering = True
        

        if self.normalization not in ['None', 'mean/var', 'Copula']:
            raise ValueError(self.normalization)

        if surrogate_model_type == 'HIERARCHICAL_KRG':
            #find the largest x from existing data:
            ds_len = 0
            len_per_task = [0]*len(training_data)
            #print(len_per_task)
            i=0
            for task in training_data:
                arr = training_data[task]['configurations']
                len_per_task[i] = arr.shape[1]
                i+=1
                if arr.shape[1]>ds_len:
                    ds_len = arr.shape[1]
                    largest_task = task

            ds_cat_list = [ str(id) for id in range(0,ds_len) ]
            dslist = [CategoricalVariable(
                ds_cat_list
            )]
            #print(ds_cat_list)
            #ds_float_list = []
            for id in range(0,ds_len):
                arr = training_data[largest_task]['configurations']
                dslist.append(FloatVariable(min(arr[:,id]),max(arr[:,id]))) 
            ds = DesignSpace(
                                dslist                                
                            )
            #print(ds)
            for id in range(3,ds_len+1):
                #tried to have decreed variables for all metavalues below id but did not work
                #for nid in range(1,id):
                #    print('decreed var', id)
                #    print('meta value', str(id-1))
                #    ds.declare_decreed_var(decreed_var=nid, meta_var=0, meta_value=str(id-1))
                ds.declare_decreed_var(decreed_var=id, meta_var=0, meta_value=str(id-1))
            
        base_models = []
        i=0
        for task in training_data: 
            if surrogate_model_type == 'HIERARCHICAL_KRG':
                model = MixedIntegerKrigingModel(
                        surrogate=KRG(
                            design_space=ds,
                            categorical_kernel=MixIntKernelType.GOWER,
                            hierarchical_kernel=MixHrcKernelType.ALG_KERNEL,
                            #theta0=[1e-2],
                            corr="abs_exp",
                            n_start=5,
                        ),
                    )
                model.options['print_global'] = False
            else: 
                model = KPLS()
                model.options['print_global'] = False
            Y = training_data[task]['y']            
            
            X = training_data[task]['configurations'] #Ali: for now assume configurations in training_data already have the vector values
            if surrogate_model_type == 'HIERARCHICAL_KRG':
                categorical_column = (len_per_task[i]-1)*np.atleast_2d(np.ones(len(X[:,0]))).T
                X = np.hstack((categorical_column,X))
                for j in range(len_per_task[i],ds_len):
                    not_acting = np.atleast_2d(np.zeros(len(X[:,0]))).T
                    X = np.hstack((X,not_acting))
            i+=1
           

            model.set_training_values(X, Y) #using SMT method
            model.train()            
            base_models.append(model)
        self.base_models = base_models
        self.surrogate_model_type = surrogate_model_type
        if surrogate_model_type == 'HIERARCHICAL_KRG': self.len_per_task = len_per_task
        self.weights_over_time = []
    
    def _model_ops(self, X: np.ndarray, Y: np.ndarray):
        res = {}
        for model_idx, model in enumerate(self.base_models):
            cfs_0 = np.array([1,  0])#1, 0, first start with a and d only 
            error_fun = lambda cfs: np.sum(np.abs(cfs[0]*model.predict_values(X).flatten() + cfs[1] - Y.flatten()))
            res[model_idx] = minimize(lambda cfs: error_fun(cfs), cfs_0, method='SLSQP',options = {'maxiter' :500})#SLSQP
            #print('comparing models with Y')
            #print(model.predict_values(X))
            #print(Y)
            #print(res[model_idx].x[0])
            #print(res[model_idx].x[1])
            #print(np.sum(np.abs(res[model_idx].x[0]*model.predict_values(X).flatten() + res[model_idx].x[1] - Y.flatten())))
            #print('results  for', model_idx, 'are', res[model_idx])
            
             # define error function
            # optimize a,b,c,d to minimize error
        return self 
    
    def _train(self, X: np.ndarray, Y: np.ndarray): #-> AbstractEPM
        if self.normalization == 'mean/var':
            Y = Y.flatten()
            mean = Y.mean()
            std = Y.std()
            if std == 0:
                std = 1

            y_scaled = (Y - mean) / std
            self.Y_std_ = std
            self.Y_mean_ = mean
        elif self.normalization in ['None', 'Copula']:
            self.Y_mean_ = 0.
            self.Y_std_ = 1.
            y_scaled = Y
            if self.normalization == 'Copula':
                y_scaled = copula_transform(Y)
        else:
            raise ValueError(self.normalization)

        #target_model = get_gaussian_process(
        #    bounds=self.bounds,
        #    types=self.types,
        #    configspace=self.configspace,
        #    rng=self.rng,
        #    kernel=None,
        #)
        target_model = KPLS()
        target_model.options['print_global'] = False
        target_model.set_training_values(X, Y) #using SMT method     

        self.target_model = target_model.train()
        
        self.model_list_ = self.base_models + [target_model]
        




        discordant_pairs_per_task = {}
        #print('number of models:')
        #print(len(self.base_models))
        res = {}
        coefficients =  np.zeros((len(self.model_list_),2))
        coefficients[:,0] = 1.0    
        #first scale models:
        if self.gp_scaling:
            for model_idx, model in enumerate(self.base_models):
                if X.shape[0] < 2:
                    weights[model_idx] = 0.75
                else:
                    
                    if self.surrogate_model_type == 'HIERARCHICAL_KRG':
                        categorical_column = (self.len_per_task[model_idx]-1)*np.atleast_2d(np.ones(len(X[:,0]))).T
                        #print(self.len_per_task[model_idx])
                        #print(categorical_column)
                        Xt = np.hstack((categorical_column,X))
                        #print(Xt)
                        if len(X[0,:])<max(self.len_per_task): # if trust data is smaller dimension than training data
                            for j in range(len(X[0,:]),max(self.len_per_task)):
                                not_acting = np.atleast_2d(np.zeros(len(Xt[:,0]))).T
                                Xt = np.hstack((Xt,not_acting))
                    else: Xt = X
                    #print(Xt)
                    #print(model.predict_values(X).shape)
                    cfs_0 = np.array([1,  0])#1, 0, first start with a and d only 
                    error_fun = lambda cfs: np.sum(np.abs(cfs[0]*model.predict_values(Xt).flatten() + cfs[1] - Y.flatten()))
                    res[model_idx] = minimize(lambda cfs: error_fun(cfs), cfs_0, method='SLSQP',options = {'maxiter' :100})#SLSQP
                    coefficients[model_idx] = res[model_idx].x

        self.coefs_ = coefficients # publish calculated coeficients
        # initialize MOE for base models
        n_clusters = self.n_clusters
        if self.clustering:
            moe = MOE(n_clusters=n_clusters,smooth_recombination=self.smooth_recombination, input_models=self.base_models)
            moe.set_training_values(X, Y)
            moe.train()
            #split target data into clusters:
            clustered_doe = moe._split_data_to_clusters()
        #then calculate weights:
        weights = np.zeros((len(self.model_list_),n_clusters))
        if self.ens_kernel == 'EPAN' or self.ens_kernel == 'EPAN2':
            weights[-1,:] = 0.75 #weight of target model to max
        elif self.ens_kernel == 'GAUS':
            weights[-1,:] = 1/(2*np.pi)**0.5
        elif self.ens_kernel == 'TRI-W':
            weights[-1,:] = 0.75#35/32
        else: print('Kernel selection is incorrect!')
        coefficients[-1,0]=1.0 #coef of target model to max
        
        for model_idx, model in enumerate(self.base_models):
            if X.shape[0] < 2: # not enough data to generate weights and coefs
                if self.ens_kernel == 'EPAN' or self.ens_kernel == 'EPAN2':
                    weights[model_idx,:] = 0.75 # if only one model, set weight to max
                elif self.ens_kernel == 'GAUS':
                    weights[model_idx,:] = 1/(2*np.pi)**0.5 # if only one model, set weight to max
                elif self.ens_kernel == 'TRI-W':
                    weights[model_idx,:] = 0.75#35/32 # if only one model, set weight to max
                #weights[model_idx,:] = 0.75 
                coefficients[model_idx,:]=1.0
            else:                
                if self.clustering:
                    # weights matrix for clusters
                    print('clustering weight calculation')
                    for c in range(n_clusters):                        
                        #calculate the weights of each model
                        doe =  np.array(clustered_doe[c])
                        nx = moe.ndim
                        xcluster = doe[:,0:nx]
                        ycluster = doe[:,nx :nx+1]# in this work, only 1 output is assumed
                        mean = coefficients[model_idx,0]*model.predict_values(xcluster) + coefficients[model_idx,1]#, _ SMT predicts variances using a different function
                        variances = coefficients[model_idx,0]*model.predict_variances(xcluster)# + coefficients[model_idx,1]#, _ SMT predicts variances using a different function
                        if self.weighting_method in ["None", "DISC-PRS"]:
                            discordant_pairs = 0
                            total_pairs = 0
                            for i in range(xcluster.shape[0]):
                                for j in range(i + 1, xcluster.shape[0]):
                                    if (ycluster[i] < ycluster[j]) ^ (mean[i] < mean[j]):
                                        discordant_pairs += 1
                                    total_pairs += 1
                            t = discordant_pairs / total_pairs / self.bandwidth
                            discordant_pairs_per_task[model_idx] = discordant_pairs
                            #alpha_t = 1 #importance of function shape in scoring
                            #alpha_a = 1 #importance of function accuracy in scoring
                            #alpha_s = 1 #importance of function variance in scoring
                            #min_acc = 0.1
                            max_var = self.max_vars*np.max(abs(mean))
                            #mse = ((Y - mean)**2).mean(axis=0)/np.max(Y)
                            #mse = abs((ycluster-mean.ravel())/ycluster)
                            mse = abs((ycluster.flatten()-mean.ravel())/ycluster.flatten())
                            
                            a = len(mse[abs(mse)>self.min_accuracy])/len(mse)/self.bandwidtha
                            s = len(variances[abs(variances)>max_var])/len(variances)/self.bandwidths
                            print('mse is ', mse)
                            print('t is', t)
                            print('a is', a)
                            print('s is', s)

                            tas  = self.alpha_t*t + self.alpha_a*a + self.alpha_s*s
                            print('mse is ', mse)
                            print('t is', t)
                            print('a is', a)
                            print('s is', s)
                            print('tas is', tas)
                            if self.ens_kernel == 'EPAN':
                                if (tas < 1) :  # The paper says <=, but the code says < (https://github.com/wistuba/TST/blob/master/src/de/ismll/hylap/surrogateModel/TwoStageSurrogate.java)
                                    weights[model_idx,c] = 0.75 * (1 - tas ** 2)
                                else:
                                    weights[model_idx,c] = 0
                            elif self.ens_kernel == 'EPAN2':
                                weights[model_idx,c] = 0
                                if (t < 1) :  
                                    weights[model_idx,c] = weights[model_idx,c] + self.alpha_t*0.75 * (1 - t** 2)
                                if (a < 1) :  
                                    weights[model_idx,c] = weights[model_idx,c] + self.alpha_a*0.75 * (1 - a** 2)
                                if (s < 1) :  
                                    weights[model_idx,c] = weights[model_idx,c] + self.alpha_s*0.75 * (1 - s** 2)                                
                            elif self.ens_kernel == 'GAUS':
                                weights[model_idx,c] = self.alpha_t*1/(2*np.pi)**0.5 * np.exp(-((t) ** 2)/2) + self.alpha_a*1/(2*np.pi)**0.5 * np.exp(-((a) ** 2)/2) + self.alpha_s**1/(2*np.pi)**0.5 * np.exp(-((s) ** 2)/2)
                            elif self.ens_kernel == 'TRI-W':
                                weights[model_idx,c] = 0
                                if (t < 1) :  
                                    weights[model_idx,c] = weights[model_idx,c] + self.alpha_t*35/32 * (1 - t** 2)**3 
                                if (a < 1) :  
                                    weights[model_idx,c] = weights[model_idx,c] + self.alpha_a*35/32 * (1 - a** 2)**3 
                                if (s < 1) :  
                                    weights[model_idx,c] = weights[model_idx,c] + self.alpha_s*35/32 * (1 - s** 2)**3 
                            
                            

                        elif (self.weighting_method == "E-REL"):                            
                            mse = ((ycluster - mean)**2).mean(axis=0)
                            weights[model_idx,c] = 1/mse**2
                        

                        #print('cluster x values are', xcluster)
                        print('model index ', model_idx, ',cluster is ', c, ',calculated t', t, ',weight ', weights[model_idx,c] )
                else:
                    discordant_pairs = 0
                    total_pairs = 0
                    Xt = X
                    if self.surrogate_model_type == 'HIERARCHICAL_KRG':
                        categorical_column = (self.len_per_task[model_idx]-1)*np.atleast_2d(np.ones(len(X[:,0]))).T
                        #print(self.len_per_task[model_idx])
                        #print(categorical_column)
                        Xt = np.hstack((categorical_column,X))
                        #print(Xt)
                        if len(X[0,:])<max(self.len_per_task): # if trust data is smaller dimension than training data
                            for j in range(len(X[0,:]),max(self.len_per_task)):
                                not_acting = np.atleast_2d(np.zeros(len(Xt[:,0]))).T
                                Xt = np.hstack((Xt,not_acting))
                    mean = coefficients[model_idx,0]*model.predict_values(Xt) + coefficients[model_idx,1]#, _ SMT predicts variances using a different function
                    variances = coefficients[model_idx,0]*model.predict_variances(Xt)# + coefficients[model_idx,1]#, _ SMT predicts variances using a different function

                    for i in range(Xt.shape[0]):
                        for j in range(i + 1, Xt.shape[0]):
                            if (Y[i] < Y[j]) ^ (mean[i] < mean[j]):
                                discordant_pairs += 1
                            total_pairs += 1
                    t = discordant_pairs / total_pairs / self.bandwidth
                    discordant_pairs_per_task[model_idx] = discordant_pairs
                    
                    max_var = self.max_vars*np.max(abs(mean))
                    #print('mean is', mean.ravel())
                    #print('Y is', Y)
                    #mse = ((Y - mean)**2).mean(axis=0)#/np.max(Y)
                    mse = abs((Y-mean.ravel())/Y)
                    a = len(mse[abs(mse)>self.min_accuracy])/len(mse)/self.bandwidtha
                    s = len(variances[abs(variances)>max_var])/len(variances)/self.bandwidths
                    #print('mse is ', mse)
                    #print('t is', t)
                    #print('a is', a)
                    #print('s is', s)

                    tas  = self.alpha_t*t + self.alpha_a*a + self.alpha_s*s

                    if self.ens_kernel == 'EPAN':
                        if (tas < 1) :  # The paper says <=, but the code says < (https://github.com/wistuba/TST/blob/master/src/de/ismll/hylap/surrogateModel/TwoStageSurrogate.java)
                            weights[model_idx] = 0.75 * (1 - tas ** 2)
                        else:
                            weights[model_idx] = 0
                    elif self.ens_kernel == 'EPAN2':
                        weights[model_idx] = 0
                        if (t < 1) :  
                            weights[model_idx] = weights[model_idx] + self.alpha_t*0.75 * (1 - t** 2)
                        if (a < 1) :  
                            weights[model_idx] = weights[model_idx] + self.alpha_a*0.75 * (1 - a** 2)
                        if (s < 1) :  
                            weights[model_idx] = weights[model_idx] + self.alpha_s*0.75 * (1 - s** 2)
                    elif self.ens_kernel == 'GAUS':
                        weights[model_idx] = self.alpha_t*1/(2*np.pi)**0.5 * np.exp(-((t) ** 2)/2) + self.alpha_a*1/(2*np.pi)**0.5 * np.exp(-((a) ** 2)/2) + self.alpha_s**1/(2*np.pi)**0.5 * np.exp(-((s) ** 2)/2)
                    elif self.ens_kernel == 'TRI-W':
                        weights[model_idx] = 0
                        if (t < 1) :  
                            weights[model_idx] = weights[model_idx] + self.alpha_t*35/32 * (1 - t** 2)**3 
                        if (a < 1) :  
                            weights[model_idx] = weights[model_idx] + self.alpha_a*35/32 * (1 - a** 2)**3 
                        if (s < 1) :  
                            weights[model_idx] = weights[model_idx] + self.alpha_s*35/32 * (1 - s** 2)**3 

                    
                    #if t < 1:  # The paper says <=, but the code says < (https://github.com/wistuba/TST/blob/master/src/de/ismll/hylap/surrogateModel/TwoStageSurrogate.java)
                    #    weights[model_idx] = 0.75 * (1 - t ** 2)
                        #print('model weight',weights[model_idx,0])                    
                    #else:
                    #    weights[model_idx,0] = 0
        #print('GP OPS Weights:',weights)
        # perform model pruning
        # use this only for ablation
        if X.shape[0] >= 2:
            p_drop = []
            if self.weight_dilution_strategy in ['probabilistic', 'probabilistic-ld']:
                for i in range(len(self.base_models)):
                    concordant_pairs = total_pairs - discordant_pairs_per_task[i]
                    proba_keep = concordant_pairs / total_pairs
                    if self.weight_dilution_strategy == 'probabilistic-ld':
                        proba_keep = proba_keep * (1 - len(X) / float(self.number_of_function_evaluations))
                    proba_drop = 1 - proba_keep
                    p_drop.append(proba_drop)
                    r = self.rng.rand()
                    if r < proba_drop:
                        weights[i] = 0
            elif self.weight_dilution_strategy == 'None':
                pass
            else:
                raise ValueError(self.weight_dilution_strategy)
        #print('weights before hard', weights)
        for i in range(n_clusters):

            if (self.ignore_target_model == True) and (len(weights[:,i])>1) and (np.sum(weights[:,i])>0.75): # Ali, this if statement added to remove impact of target model if more than 1 base model is assigned a weight and 
                weights[-1,i] = 0

            if self.smooth_weights==True:
                weights[:,i] /= np.sum(weights[:,i])
            else: 
                for j, w in enumerate(weights[:,i]):
                    #print('w is', w)
                    #print('max weights is ',np.max(weights[:,i]))
                    if w == np.max(weights[:,i]):
                        weights[j,i] = 1.0
                    else: 
                        weights[j,i] = 0.0
                        #print('weights after hard', weights)
       
            low_weights = 0.0001 # assigning a low weight value to prevent error in case none of the source models are higher than 0
            if (np.sum(weights[:,i])<low_weights): 
                print('Warning: none of the source models have none zero weight. Assigning an equal weight to all source models')
                weights[0:-1, i] = 1/len(weights[0:-1, i])
        #print('GP OPS Weights:')
        #print(weights)
        self.weights_ = weights
        if self.clustering:
            self.moe_ = moe
        

        self.weights_over_time.append(weights)
        # create model and acquisition function
        return self

    def _predict(self, X: np.ndarray, cov_return_type: str = 'diagonal_cov') -> Tuple[np.ndarray, np.ndarray]:

        if cov_return_type != 'diagonal_cov':
            raise NotImplementedError(cov_return_type)
        probs = np.ones((np.shape(np.array(X))[0],int(self.n_clusters)))
        if self.clustering:
            distribs = self.moe_.distribs
            
            probs = np.array(
                [self.moe_._proba_cluster_one_sample(row,distribs) for row in np.atleast_2d(X)]
            )
            #print('x probabilities in clusters are', probs)
        # compute posterior for each model
        weighted_means = []
        weighted_covars = []
        #print('input weights are ', self.weights_)

        for c in range(self.n_clusters):


            # filter model with zero weights
            # weights on covariance matrices are weight**2
            non_zero_weight_indices = (self.weights_[:,c] ** 2 > 0).nonzero()[0]
            non_zero_weights = self.weights_[non_zero_weight_indices,c]
            # re-normalize
            non_zero_weights /= non_zero_weights.sum()
            #print('no zero weights for cluster ',c, ' are ', non_zero_weights)


            for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
                raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
                weight = non_zero_weights[non_zero_weight_idx]
                #print('model:', raw_idx, 'coef 0:', self.coefs_[raw_idx,0], 'model object',self.model_list_[raw_idx] )
                if self.surrogate_model_type == 'HIERARCHICAL_KRG':
                    categorical_column = (self.len_per_task[raw_idx]-1)*np.atleast_2d(np.ones(len(X[:,0]))).T
                    #print(self.len_per_task[raw_idx])
                    #print(categorical_column)
                    Xt = np.hstack((categorical_column,X))
                    
                    if len(X[0,:])<max(self.len_per_task): # if trust data is smaller dimension than training data
                        for j in range(len(X[0,:]),max(self.len_per_task)):
                            not_acting = np.atleast_2d(np.zeros(len(Xt[:,0]))).T
                            Xt = np.hstack((Xt,not_acting))
                    prediction = self.model_list_[raw_idx].predict_values(Xt)[:,0]
                    mean = self.coefs_[raw_idx,0]*prediction + self.coefs_[raw_idx,1]
                #print(mean)
                    
                    covar = self.coefs_[raw_idx,0]*self.model_list_[raw_idx].predict_variances(Xt)[:,0]#+ self.coefs_[raw_idx,1]
                else: 
                    mean = self.coefs_[raw_idx,0]*self.model_list_[raw_idx]._predict_values(X) + self.coefs_[raw_idx,1]
                #print(mean)
                    covar = self.coefs_[raw_idx,0]*self.model_list_[raw_idx]._predict_variances(X)# + self.coefs_[raw_idx,1]# SMT predicts convariances separately
                #print(covar)
                #print('weight is', weight)
                #print('mean is', mean)
                #print('probability is', probs[:,c])

                weighted_means.append(weight * mean * probs[:,c])

                if self.variance_mode == 'average':
                    weighted_covars.append(covar * weight ** 2)
                elif self.variance_mode == 'target':
                    if raw_idx + 1 == len(self.weights_):
                        weighted_covars.append(covar)
                else:
                    raise ValueError()

            if len(weighted_covars) == 0:
                if self.variance_mode != 'target':
                    raise ValueError(self.variance_mode)
                #_, covar = self.model_list_[-1]._predict(X, cov_return_type)
                #covar = self.model_list_[-1]._predict_variances(X)
                #TO DO: fix covariance here, updated temporarily from -1 to -2 to ignore target model
                #covar = self.model_list_[-2]._predict_variances(X)
                weighted_covars.append(covar)

        # set mean and covariance to be the rank-weighted sum the means and covariances
        # of the base models and target model
#        mean_x = np.sum(np.stack(weighted_means), axis=0) * self.Y_std_ + self.Y_mean_
        #print('weighted means are', weighted_means)
        mean_x = np.sum(np.stack(weighted_means), axis=0)
#        covar_x = np.sum(weighted_covars, axis=0) * (self.Y_std_ ** 2)
        covar_x = np.sum(weighted_covars, axis=0)
        return mean_x, covar_x

    def sample_functions(self, X_test: np.ndarray, n_funcs: int=1) -> np.ndarray:
        """
        Samples F function values from the current posterior at the N
        specified test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        n_funcs: int
            Number of function values that are drawn at each test point.

        Returns
        ----------
        function_samples: np.array(F, N)
            The F function values drawn at the N test points.
        """

        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights_ ** 2 > 0).nonzero()[0]
        non_zero_weights = self.weights_[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()

        samples = []
        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            weight = non_zero_weights[non_zero_weight_idx]

            funcs = self.model_list_[raw_idx].sample_functions(X_test, n_funcs)
            funcs = funcs * weight
            samples.append(funcs)
        samples = np.sum(samples, axis=0)
        return samples
