import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit

class Model1():
    """
    fits a sigmoid curve to the choice data
    """
    
    def __init__(self, data):
        """
        """
        
        self.data = data
        self._f = lambda x,a,b: 1 / (1 + np.e ** -(a - (b * x)))
        
    def fit(self, iv='x_light'):
        """
        """
        
        #
        self.iv = iv
        data0 = self.data[self.data[iv]==0]
        data1 = self.data[self.data[iv]==1]
        
        # 
        X0 = np.array(data0['x_frac_odor_left'],dtype=float)
        X1 = np.array(data1['x_frac_odor_left'],dtype=float)
        y0 = np.array(data0['x_choice'],dtype=int)
        y1 = np.array(data1['x_choice'],dtype=int)
        
        # fit.
        (a0,b0), pcov = curve_fit(self._f, X0, y0, maxfev=10000)
        (a1,b1), pcov = curve_fit(self._f, X1, y1, maxfev=10000)
            
        self.params0 = (a0,b0)
        self.params1 = (a1,b1)
        
        # compute the difference in bias
        bias0 = ((a0 / b0) + 0.5) * 100
        bias1 = ((a1 / b1) + 0.5) * 100
        self.bias = bias0 - bias1
        
        return self.bias
               
    def shuffle_and_refit(self):
        """
        """
        
        # shuffle.
        data = self.data.copy()
        data[self.iv] = np.random.permutation(data[self.iv].values)
        data0 = data[data[self.iv]==0]
        data1 = data[data[self.iv]==1]
        
        # 
        X0 = np.array(data0['x_frac_odor_left'])
        X1 = np.array(data1['x_frac_odor_left'])
        y0 = np.array(data0['x_choice'],dtype=int)
        y1 = np.array(data1['x_choice'],dtype=int)
        
        # fit.
        try:
            (a0,b0), pcov = curve_fit(self._f, X0, y0,maxfev=10000)
            (a1,b1), pcov = curve_fit(self._f, X1, y1,maxfev=10000)
            
        except:
            return np.nan
        
        # compute light effect.
        bias0 = ((a0 / b0) + 0.5) * 100
        bias1 = ((a1 / b1) + 0.5) * 100
        
        return bias0 - bias1
    
    def draw(self, n=101):
        """
        """

        x = np.linspace(0,1,n)
        y0 = self._f(x,*self.params0)
        y1 = self._f(x,*self.params1)
        
        return (x,y0,y1)
    
class HomebrewLogisticRegression():
    """
    this model is based on the following online tutorial:
    
    https://beckernick.github.io/logistic-regression-from-scratch/ 
    
    I coded this just to make sure I understand how logistic regression works and to verify
    the coefficient estimates produced by the other implementations of logistic regression
    are reproducible.
    """
    
    def __init__(self):
        """
        """
        
        # attributes.
        self.data = None
        self.iv = None
        self.weights = None
    
    def link(self, eta):
        """
        link function
        """
        
        y = 1 / (1 + np.e * (np.exp(-eta)))
        
        return y
    
    def logliklihood(self, X, y, weights):
        """
        log-likelihood
        """
        
        eta = np.dot(X, weights)
        ll = np.sum(y * eta - np.log(1 + np.exp(eta)))
        
        return ll
    
    def fit(self, X, y, tol=1e-10, n_iters=5000, lr=1e-2, add_intercept=True):
        """
        logistic regression
        
        keywords
        --------
        data : pandas.DataFrame
            standard data form returned by the afcpy.analysis.mat_to_data function
        iv : str (default is 'x_light')
            independent variable
        tol : float (default is 0.0000000001)
            minimum gain in log-likelihood which indicates successful parameter estimation
        n_iters : int (default is 5000)
            maximum number of iterations performed for maximizing the log-likelihood function
        lr : float (default is 0.01)
            learning rate for gradient ascent algorithm
        add_intercept : bool (default is True)
            if True an intercept term is estimated
        update_attr : bool (default is True)
            if True the weights and bias attributes will be overwritten
            
        returns
        -------
        weights : list
            list of estimated parameters (i.e. coefficients)
        """
        
        self.X = X
        self.y = y
        
        if add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack([X,intercept])
            
        weights = np.zeros(X.shape[1])
        
        ll_prev_iter = self.logliklihood(X,y,weights)
        for i in range(n_iters):
            
            # move down gradient and recalculate weights.
            eta = np.dot(X, weights)
            predictions = self.link(eta)
            error = y - predictions
            gradient = np.dot(X.T, error)
            weights += lr * gradient
            
            # evaluate log-likelihood function.
            ll = self.loglikelihood(X,y,weights)
            gain = abs(ll - ll_prev_iter)
            if gain <= tol:
                print('tolerance of {} reached.'.format(tol))
                break
            ll_prev_iter = ll
            
        return weights
        
class FirthLogisticRegression():
    """
    This model performs logistic regression using Firth's penalty. The code is mostly based on
    the github gist link [1] provided in the sources.
    
    sources
    -------
    [1] https://gist.github.com/johnlees/3e06380965f367e4894ea20fbae2b90d
    [2] https://www.ncbi.nlm.nih.gov/pubmed/12758140
    [3] https://academic.oup.com/biomet/article-abstract/80/1/27/228364?redirectedFrom=fulltext
    """
    
    def __init__(self):
        """
        """
        
        return
    
    def likelihood(self, beta, logit):
        """
        """
        
        return -(logit.loglike(beta) + 0.5*np.log(np.linalg.det(-logit.hessian(beta))))
    
    def fit(self, y, X, start_vec=None, step_limit=1000, convergence_limit=0.0001):
        """
        """

        logit_model = Logit(y, X)
        
        if start_vec is None:
            start_vec = np.zeros(X.shape[1])
        
        beta_iterations = []
        beta_iterations.append(start_vec)
        for i in range(0, step_limit):
            pi = logit_model.predict(beta_iterations[i])
            W = np.diagflat(np.multiply(pi, 1-pi))
            var_covar_mat = np.linalg.pinv(-logit_model.hessian(beta_iterations[i]))
    
            # build hat matrix
            rootW = np.sqrt(W)
            H = np.dot(np.transpose(X), np.transpose(rootW))
            H = np.matmul(var_covar_mat, H)
            H = np.matmul(np.dot(rootW, X), H)
    
            # penalised score
            U = np.matmul(np.transpose(X), y - pi + np.multiply(np.diagonal(H), 0.5 - pi))
            new_beta = beta_iterations[i] + np.matmul(var_covar_mat, U)
    
            # step halving
            j = 0
            while self.likelihood(new_beta, logit_model) > self.likelihood(beta_iterations[i], logit_model):
                new_beta = beta_iterations[i] + 0.5*(new_beta - beta_iterations[i])
                j = j + 1
                if (j > step_limit):
                    sys.stderr.write('Firth regression failed\n')
                    return None
    
            beta_iterations.append(new_beta)
            if i > 0 and (np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) < convergence_limit):
                break
    
        return_fit = None
        if np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) >= convergence_limit:
            sys.stderr.write('Firth regression failed\n')
        else:
            # Calculate stats
            fitll = -self.likelihood(beta_iterations[-1], logit_model)
            intercept = beta_iterations[-1][0]
            beta = beta_iterations[-1][1:].tolist()
            bse = np.sqrt(np.diagonal(-logit_model.hessian(beta_iterations[-1])))
            
            return_fit = intercept, beta, bse, fitll
    
        return return_fit
        
class Model2():
    """
    models an animal's decision making using logistic regression
    
    keywords
    --------
    data : pandas.DataFrame
        standard form of task performance (see afcpy.analysis.mat_to_data)
        
    notes
    -----
    There are several available backends to choose from for performing the regression.
    I recommend using sci-kit learn's LogisticRegression class because it implements
    regularization. This prevents the over-fitting/-estimation of the beta coefficients.
    """
    
    def __init__(self, data, **model_props):
        """
        """
        
        _model_props = {'method':'bfgs',    # for statsmodels
                        'solver':'lbfgs',   # for sklearn
                        'maxiter':150,      # for all backends
                        'disp':0,           # for statsmodels
                        'C':1.0,            # for sklearn
                        }
        
        for prop in model_props.keys():
            if prop in _model_props.keys():
                _model_props[prop] = model_props[prop]
                
        self.model_props = _model_props
        self.fitted = False
        self.data = data
    
    def fit_with_sklearn(self, iv='x_light', dv='x_choice'):
        """
        fits the choice data with sci-kit learn's LogisticRegression class
        """
        
        self.iv = iv
        self.ivs = ['x_odor_left','x_odor_right',self.iv]
        self.X = self.data.loc[:,self.ivs].astype(float)
        self.y = self.data.loc[:,dv].astype(int)
        
        self.model = LogisticRegression(solver=self.model_props['solver'],
                                        max_iter=self.model_props['maxiter'],
                                        C=self.model_props['C']
                                        )
        self.model.fit(self.X,self.y)
        self.coefs = {'x_odor_left':self.model.coef_.flatten()[0],
                      'x_odor_right':self.model.coef_.flatten()[1],
                      iv:self.model.coef_.flatten()[-1] * -1,
                      'x_intercept':self.model.intercept_.item()
                      }
        
        self.bias = self.coefs[self.iv]
        self.fitted = True
        
        return self.bias
    
    def fit_with_sm(self, iv='x_light', dv='x_choice'):
        """
        fits the choice data with statsmodels Logit class
        
        returns
        -------
        bias : float
            the estimate of the coefficient for the target independent variable
        """
        
        self.iv = iv
        self.ivs = ['x_odor_left','x_odor_right',self.iv]
        self.X = self.data.loc[:,self.ivs].astype(float)
        self.y = self.data.loc[:,dv].astype(int)
        
        # use statsmodels Logit class
        self.X['x_intercept'] = 1.0 # add an empty column for the intercept term
        self.model = Logit(self.y,self.X)
        self.results = self.model.fit(method=self.model_props['method'],
                                      maxiter=self.model_props['maxiter'],
                                      disp=self.model_props['disp']
                                      )
        
        self.coefs = {'x_odor_left':self.results.params['x_odor_left'],
                      'x_odor_right':self.results.params['x_odor_right'],
                      iv:self.results.params[iv] * -1,
                      'x_intercept':self.results.params['x_intercept']
                      }
        
        self.bias = self.coefs[self.iv]
        self.fitted = True
        
        return self.bias
    
    def fit_with_firth(self, iv='x_light', dv='x_choice'):
        """
        applies Firth's penalty to the maximum likelihood estimation
        """
        
        self.iv = iv
        self.ivs = ['x_odor_left','x_odor_right',self.iv]
        self.X = self.data.loc[:,self.ivs].astype(float)
        self.X['x_intercept'] = 1.0 # add an empty column for the intercept term
        self.y = self.data.loc[:,dv].astype(int)
        
        # use firth penalty for MLE
        self.model = FirthLogisticRegression()
        intercept,betas,bse,fitl1 = self.model.fit(self.y,self.X)
        b1,b2,b3 = betas
        self.coefs = {'x_odor_left':b1,
                      'x_odor_right':b2,
                      self.iv:b3 * -1,
                      'x_intercept':intercept
                      }
        
        self.bias = self.coefs[self.iv]
        self.fitted = True
        
        return self.bias
    
    def fit_with_hb(self, iv='x_light', dv='x_choice'):
        """
        fit choice data with my homebrew logistic regression class
        """
        
        self.iv = iv
        self.ivs = ['x_odor_left','x_odor_right',self.iv]
        self.X = self.data.loc[:,self.ivs].astype(float)
        self.X['x_intercept'] = 1.0 # add an empty column for the intercept term
        self.y = self.data.loc[:,dv].astype(int)
        
        self.model = HomebrewLogisticRegression()
        b1,b2,b3,b0 = self.model.fit(X,y)
        self.coefs = {'x_odor_left':b1,
                      'x_odor_right':b2,
                      self.iv:b3 * -1,
                      'x_intercept':b0
                      }
        self.bias = self.coefs[self.iv]
        self.fitted = True

        return self.bias
    
    def fit_with_custom_ivs(self, predictors=['x_frac_odor_left','x_light'], target_predictor='x_light', outcome='x_choice'):
        """
        perform regression with an alternative set of predictor variables
        
        keywords
        --------
        predictors : list
            custom set of predictor variables
        target_predictor : str
            identifies the independent variable which denotes the experimental condition
            
        returns
        -------
        bias : float
            beta coefficient estimated for the target predictor
            
        notes
        -----
        This method currently only implements statsmodels Logit class
        """
        try:
            self.X = self.data.loc[:,predictors].astype(float)
            self.X['x_intercept'] = 1.0 # add an empty column for the intercept term
            self.y = self.data.loc[:,outcome].astype(int)
            
            self.model = Logit(self.y,self.X)
            self.results = self.model.fit(method=self.model_props['method'],
                                          maxiter=self.model_props['maxiter'],
                                          disp=self.model_props['disp']
                                          )
            
        except:
            print("ERROR: Regression failed.")
            return
        
        self.coefs = dict(zip(predictors,self.results.params))
        
        self.bias = self.coefs[target_predictor]
        self.fitted = True
        
        return self.bias
            
    def shuffle_and_refit(self):
        """
        shuffles the values in the columns of the target independent
        variable and refits the model
        
        returns
        -------
        bias : float
            estimate of the coefficient for the target independent variable
        """
        
        if not self.fitted:
            print('WARNING: The model must be fit with unshuffled data first.')
            return
        
        # shuffle the values in the column for the target independent variable
        data = self.data.copy()
        data[self.iv] = np.random.permutation(data[self.iv].values)
        X_shuf = np.array(data.loc[:,self.ivs])
        
        # refit the shuffled data
        self.model.fit(X_shuf,self.y)
        coefs = {'x_odor_left':self.model.coef_.flatten()[0],
                 'x_odor_right:None':self.model.coef_.flatten()[1],
                 self.iv:self.model.coef_.flatten()[-1] * -1,
                 'x_intercept':self.model.intercept_.item()
                 }
        
        bias = coefs[self.iv] * -1
        
        return bias
    
    def score(self, method='MSE'):
        """
        evaluates the performance of the model
        
        returns
        -------
        score : float
            mean square error
        """
        
        if not self.fitted:
            msg = 'Fit method must be called before scoring the model.'
            raise ValueError(msg)
        
        if method not in ['MSE']:
            msg = "Method must be one of ['MSE']."
            raise ValueError(msg)
        
        y_prob = self.results.predict()
        y_true = np.array(self.y)
        
        self.score = brier_score_loss(y_true, y_prob)
        
        return self.score
    
    def draw(self, n=101):
        """
        computes the probability of making a left choice across the full range
        of possible values of the stimulus intensity
        
        keywords
        --------
        n : int
            resolution of the curves (or number of points)
        
        returns
        -------
        x : numpy.ndarray
            values for the fraction of the left odor
        y0 : numpy.ndarray
            predicted probabilities given the value of the target independent variable equals 0
        y1 : numpy.ndarray
            predicted probabilities given the value of the target independent variable equals 1
        """
        
        # the sample size must be odd to include exactly 0.5 frac. of the left odor
        if n%2 == 0:
            msg = "The size of the sample must be odd."
            raise ValueError(msg)
        
        # construct an array for each predictor that spans the full range of possible values
        f = lambda x: (x - 0.5) / 0.5
        x_odor = f(np.linspace(0,1,n))
        x_odor[x_odor<0] = 0
        x_odor_left = x_odor
        x_odor_right = x_odor[::-1]
        x_iv_off = np.zeros(n) 
        x_iv_on = np.ones(n) 
        
        # stack everything
        data_iv_off = np.hstack([x_odor_left.reshape(-1,1), 
                                 x_odor_right.reshape(-1,1), 
                                 x_iv_off.reshape(-1,1)])
        data_iv_on = np.hstack([x_odor_left.reshape(-1,1), 
                                x_odor_right.reshape(-1,1), 
                                x_iv_on.reshape(-1,1)]) 
        
        # create a dataframe for each condition
        labels = ['x_odor_left','x_odor_right',self.iv]
        X_iv_off = pd.DataFrame(data_iv_off,columns=labels) 
        X_iv_on = pd.DataFrame(data_iv_on,columns=labels) 
        
        # add an intercept
        X_iv_off['x_intercept'] = self.coefs['x_intercept'] 
        X_iv_on['x_intercept'] = self.coefs['x_intercept'] 
        
        # run the arrays through the model
        x = np.linspace(0,1,n)
        y0 = self.results.predict(X_iv_off) 
        y1 = self.results.predict(X_iv_on)
       
        return (x,y0,y1)