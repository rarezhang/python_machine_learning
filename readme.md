# Python Machine Learning  

reading notes of **[Python Machine Learning](https://www.amazon.com/dp/B00YSILNL0/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)** 

## Reading notes  


### three different types of machine learning  
P 27  
- supervised learning  
    + supervised: a set of samples where the desired output signals (labels) are already known  
    + classification:  
        * predict the categorical class labels of new instances based on past observations  
        * learn a model from labeled training data --> allows to make predictions about unseen or future data  
    + regression:  
        * the outcome signal: a continuous value  
        * given a number of predictor (explanatory) variables and a continuous response variable (outcome), and we try to find a relationship between those variables that allows us to predict an outcome  
- unsupervised learning  
    + dealing with unlabeled data or data of unknown structure  
        * explore the structure of data --> discovering hidden structures  
        * extract meaningful information without the guidance of a known outcome variable or reward function  
    + clustering: organize a pile of information into meaningful subgroups (clusters) without having any prior knowledge of their group memberships  
    + dimensionality reduction:  
        * commonly used approach in feature preprocessing to remove noise from data  
        * degrade the predictive performance of certain algorithms, and compress the data onto a smaller dimensional subspace while retaining most of the relevant information  
        * for data visualization  
- reinforcement learning  
    + goal: develop a system (agent) that improves its performance based on interactions with the environment --> solving interactive problems  
    + maximizes the **_reward_** via an exploratory trial-and-error approach or deliberative planning  
        * reward signal: not the correct ground truth label or value, but a measure of **how well the action was measured** by a reward function  

        
        
### parametric V.S. non-parametric models  
P 118  
- parametric model  
    + estimate parameters from the training dataset to learn a function that can classify new data points without requiring the original training dataset anymore
        * e.g., perceptron, logistic regression, linear SVM  
    + the number of parameters grows with the training data
        * e.g., decision tree, random forest, kernel SVM, KNN   
        
        
### predictive modeling (work flow diagram)  
![work flow diagram](https://cloud.githubusercontent.com/assets/5633774/24131161/1aa33044-0daa-11e7-896b-15da846f6657.png)
- preprocessing:  
    + extract meaningful features  
    + same scale for optimal performance  
    + dimensionality reduction: features may be highly correlated  
    + separate training and test set  
- training and selecting a predictive model  
    + each classification algorithm has its inherent biases --> **no single classification model enjoys superiority**  
    + define metric to measure performance  
    + cross-validation  
    + optimization techniques: fine-tune the performance 
- evaluating models and predicting unseen data instances  
    + how well model performs on the unseen data --> generalization error  
    + predict new, future data  

    

### algorithms alternative implementations in scikit-learn !!!
- stochastic gradient descent version 
    * when datasets are too large to fit into computer memory  
    * support online learning  
    ```
    >>> from sklearn.linear_model import SGDClassifier
    >>> ppn = SGDClassifier(loss='perceptron')
    >>> lr = SGDClassifier(loss='log')
    >>> svm = SGDClassifier(loss='hinge')
    ```    
    
    
    
### perceptron  
![perceptron](https://cloud.githubusercontent.com/assets/5633774/24132180/c8886886-0db0-11e7-8885-a61f28592f56.png)  
- weight vector **_w_**, input value **_x_**, activation function **_f(z)_**  
![perceptron_1](https://cloud.githubusercontent.com/assets/5633774/24131835/32724ae4-0dae-11e7-8c04-28338c147e5b.png)  
![perceptron_2](https://cloud.githubusercontent.com/assets/5633774/24131862/5fbb9168-0dae-11e7-9a57-61ba250d1491.png)  
- update weight based on unit step function  
- learning rate η: (0, 1] --> requires experimentation to find a good learning rate  
- steps:  
    1. initialize the weights to 0 or small random numbers  
    2. for each training sample x(i) perform the following steps  
        1. compute the output value y' --> class label  
        2. update the weights: in the case of a wrong prediction, the weights are being pushed towards the direction of the positive or negative target class  
        ![perceptron_weight1](https://cloud.githubusercontent.com/assets/5633774/24132041/a09853b4-0daf-11e7-87e7-67ebfd34fb95.png)  
        ![perceptron_weight2](https://cloud.githubusercontent.com/assets/5633774/24132056/b7561de8-0daf-11e7-8ca1-2ec05d47fcae.png)
- Note: the convergence of the perceptron is only guaranteed if 
    + the two classes are linearly separable  
        * set a maximum number of passes over the training dataset (epochs)  
        * a threshold for the number of tolerated mis-classifications  
    + the learning rate is sufficiently small  
- **perceptron algorithm never converges on datasets that aren't perfectly linearly separable, which is why the use of the perceptron algorithm is typically not recommended in practice**    
    
        
### adaptive linear neurons --> ADAptive LInear NEuron (Adaline)  
![adaline](https://cloud.githubusercontent.com/assets/5633774/24161424/9a24cb30-0e21-11e7-80e6-c8148d48160c.png)  
- update weight based on a linear activation function --> the identity function of the net input  
![adaline_weight](https://cloud.githubusercontent.com/assets/5633774/24161375/7604941a-0e21-11e7-8a27-acb2b706ec9e.png)  
- quantizer: used to predict the class labels (similar to the unit step function)  



### cost function & gradient descent  
- an objective function: to be optimized during the learning process  
- e.g., sum of squared errors (SSE)  
![sse](https://cloud.githubusercontent.com/assets/5633774/24161624/35a80950-0e22-11e7-83bb-9f377c5aa1a6.png)  
    + differentiable  
    + convex  
- gradient descent: find the weights that minimize the cost function(if differentiable & convex) for classification  
![gradient descent](https://cloud.githubusercontent.com/assets/5633774/24162045/7575e3f8-0e23-11e7-8568-d012f487a3b8.png)  
![weight_up](https://cloud.githubusercontent.com/assets/5633774/24162069/8bc27aa4-0e23-11e7-9cf5-9db68aa7aa11.png)  
![cost_fun_derivative](https://cloud.githubusercontent.com/assets/5633774/24162108/ac4c93a4-0e23-11e7-96bf-53aa7f9ece1a.png)  
![update_rule](https://cloud.githubusercontent.com/assets/5633774/24162139/d2f598fc-0e23-11e7-8594-4ff36bf0ca66.png)  


### stochastic gradient descent --> large scale machine learning  
- iterative or on-line gradient descent  
- an approximation of gradient descent: does not reach the global minimum but an area very close to it  
- advantage:
    + reach convergence much faster: because of more frequent weight updates  
    + can escape shallow local minimum more readily  
    + can be used for on-line learning: immediately adapt to changes and the training data can be discarded after updating the model if storage space in an issue  
- requires: 
    + data in random order --> shuffle the training set for every epoch to prevent cycles  
    + an adaptive learning rate that decreases over time: c1 / (number_of_iterations + c2)  
- update the eights incrementally for each training sample  
![stochastic gradient descent](https://cloud.githubusercontent.com/assets/5633774/24180262/2cb92fd2-0e71-11e7-8c3b-fbdfcde05fd1.png)  


### standardization 
P 66  
- gives the data the property of a standard normal distribution  
- feature scaling for optimal performance  
- mean of each feature is centered at value **_0_** and the feature column has a standard deviation of **_1_**  
![standardization](https://cloud.githubusercontent.com/assets/5633774/24179800/ab5f2f4c-0e6e-11e7-9ff8-37c82c4bbbc5.png)  

    
### choosing a classification algorithm  
P 75  
- no single classifier works best across all possible scenarios  
- compare the performance of at least a handful of different learning algorithms to select the best model for the particular problem  
- the performance of a classifier, computational power as well as predictive power, depends heavily on the underlying data that are available for learning  
- training a machine learning algorithm  
    1. selection of features  
    2. choosing a performance metric  
    3. choosing a classifier and optimization algorithm  
    4. evaluating the performance of the model  
    5. tuning the algorithm  


### target labels --> store as integers  
- for the optimal performance of many machine learning algorithms  

    
### greedy algorithms V.S. exhaustive search algorithms  
P 144  
- greedy algorithm: make locally optimal choices at each stage of a combinatorial search problem and generally yield a suboptimal solution --> allow for a less complex solution, computationally efficient  
- exhaustive search algorithm: evaluate all possible combinations and are guaranteed to find the optimal solution --> often not computationally feasible  
    
### overfitting & underfitting  
P 90 P 137  
![under_over_fitting](https://cloud.githubusercontent.com/assets/5633774/24261285/c379a60c-0fb3-11e7-9d91-f9e4144ad923.png)  
- overfitting:  
    + definition: a model performs well on training data but does not generalize well to unseen data (test data)  
    + model has a high variance: too many parameters that lead to a model that is too  complex   
    + common solution:
        1. collect more training data  
        2. introduce a penalty for complexity via regularization  
        3. choose a simpler model with fewer parameters  
        4. reduce the dimensionality of the data  
- underfitting:  
    + model is not complex enough to capture the pattern in the training data --> low performance on unseen data  
    + model has a high bias  
- variance: measures the consistency of the model prediction (model is sensitive to the randomness)  
- bias: measure of the systematic error that is not due to randomness (how far off the predictions are from the correct values in general)  
    
    
### regularization  
P 91 P 138  
- regularization: 
    + add a penalty term to the cost function to encourage smaller weights --> penalize large weights  
    + by increasing the regularization strength via the regularization parameter λ --> shrink the weights towards zero and decrease the dependence of our model on the training data  
- tune the model via **regularization**:  
    + find a good bias-variance trade-off (model complex)  
    + handle collinearity (high correlation among features)  
- introduce additional information (bias) to penalize extreme parameter weights  
    + requirement: for regularization to work properly, need to ensure that all features are on comparable scales --> feature standardization  
    + L1 regularization:  
    ![regularization-l1](https://cloud.githubusercontent.com/assets/5633774/24333600/b39a90f8-120f-11e7-9f14-6981dfa131ae.png)  
    ![nor-l1](https://cloud.githubusercontent.com/assets/5633774/24333131/f016b23a-1207-11e7-88c1-8bfa796b0caa.png)  or  ![l1](https://cloud.githubusercontent.com/assets/5633774/24262011/ff1bfafa-0fb5-11e7-8f9c-045f9f6e2262.png)  
        * encourage **sparsity**: yields sparse feature vector --> most feature weights will be zero  
        * **sparsity** is useful in practice if dataset has high-dimension --> many features are irrelevant  
        * especially useful when there are more irrelevant dimensions than samples --> can be understand as a technique for feature selection  
    + L2 regularization:  
    ![regularization-l2](https://cloud.githubusercontent.com/assets/5633774/24333512/f6c964dc-120d-11e7-8777-b90b68140f98.png)  
    ![nor-l2](https://cloud.githubusercontent.com/assets/5633774/24333135/fe8ed982-1207-11e7-89c8-c2f914e446dc.png)  or  ![l2](https://cloud.githubusercontent.com/assets/5633774/24262041/16d22cf0-0fb6-11e7-8113-0eba38cbae3b.png)  
        


        
        
### dimensionality reduction  
P 143  
- feature selection  
- feature extraction  
- assess feature importance  


### feature selection  
P 143  
- select a subset of the original features  
- sequential feature selection: greedy search algorithms  
    + reduce an initial d-dimensional feature space to a k-dimensional feature subspace where k < d  
    + motivation: automatically select a subset of features that are most relevant to the problem:  
        1. improve computational efficiency  
        2. reduce the generalization error of the model by removing irrelevant features or noise (useful for algorithms that don't support regularization)  
    + **Sequential Backward Selection (SBS)**: sequentially removes features from the full feature subset until the new feature subspace contains the desired number of features  
        1. reduce the dimensionality of the initial feature subspace with a minimum decay in performance of the classifier to improve upon computational efficiency  
        2. can improve the predictive power of the model if a model suffers from overfitting  
    + SBS steps:
        1. initialize the algorithm with ```k=d```, where ```d``` is the dimensionality of the full feature space ```Xd```  
        2. determine the feature ```x'``` that maximizes the criterion ```x'= argmax C(Xk-x)``` where x belongs to Xk  
        3. remove the feature ```x'``` from the feature set  
        4. terminate if ```k``` equals the number of desired features; if not, go to step 2  
            
            
            
### feature extraction  
P 152  
- derive information from the feature set to construct a new feature subspace --> summarize the information content of a dataset by transforming it onto a new feature subspace of lower dimensionality than the original one  
    + data compression with the goal of maintaining most of the relevant information  
    + improve computational efficiency  
    + reduce the curse of dimensionality (especially with non-regularized models)  
- principal component analysis (**PCA**):  
    + for **unsupervised** data compression  
    + identify patterns in data based on the correlation between features  
    + aims to find the directions of maximum variance in high-dimensional data and projects it onto a new subspace with equal or fewer dimensions  
    + the orthogonal axes (principal components) of the new subspace: the directions of maximum variance given the constraint that the new feature axes are orthogonal to each other as illustrated  
    ![pca](https://cloud.githubusercontent.com/assets/5633774/24339387/2b2fde20-1260-11e7-890c-798a88eb6204.png)  
        1. standardize the **d**-dimensional dataset  
        2. construct the covariance matrix  
        ![covariance_matrix](https://cloud.githubusercontent.com/assets/5633774/24339879/59234d1e-1263-11e7-988f-43be5546a82e.png)  
        positive covariance between two features indicates that the features increase or decrease together, whereas a negative covariance indicates that the features vary in opposite directions  
        3. decompose the covariance matrix into its eigenvectors and eigenvalues  
        4. select **k** eigenvectors that correspond to the **k** largest eigenvalues --> **k**: the dimensionality of the new feature subspace (k<=d)  
        5. construct a projection matrix **w** from the top **k** eigenvectors  
        6. transform the **d**-dimensional input dataset X using the projection matrix **W** to obtain the new k-dimensional feature subspace  
- linear discriminant analysis (**LDA**):  
    + **supervised** dimensionality reduction technique for maximizing class separability  
    + find the feature subspace that optimizes class separability  
    + assumptions:  
        1. data is normally distributed  
        2. the classes have identical covariance matrices  
        3. features are statistically independent of each other  
        * Note: even if one or more of those assumptions are slightly violated, LDA for dimensionality reduction can still work reasonably well  
    + key steps:
        1. standardize the ```d```-dimensional dataset (```d```: number of features)  
        2. for each class, compute the ```d```-dimensional mean vector  
        ![mean_vector](https://cloud.githubusercontent.com/assets/5633774/24369004/7634f046-12d7-11e7-8e09-91c853e31449.png)  
        3. construct the between-class scatter matrix ```Sb``` and the within-class scatter matrix ```Sw```  
        individual scatter matrices:  
        ![individual_scatter_matrix](https://cloud.githubusercontent.com/assets/5633774/24369372/a10415a8-12d8-11e7-87db-4c553c834878.png)  
        within-class scatter matrix:  
        ![within-class scatter matrix](https://cloud.githubusercontent.com/assets/5633774/24369439/d42d738e-12d8-11e7-81d8-133ab65f6332.png)  
        normalized scatter matrix:  
        ![normalized scatter matrix](https://cloud.githubusercontent.com/assets/5633774/24369825/1ac72ea6-12da-11e7-897c-f16dc63f1db2.png)  
        between class scatter matrix:  
        ![between-class scatter matrix](https://cloud.githubusercontent.com/assets/5633774/24370291/c3ef2c76-12db-11e7-8f7b-39c4410c7e03.png)  
        4. compute the eigenvectors and corresponding eigenvalues of the matrix ```(Sw)^(-1)*(Sb)```  
        5. choose the k eigenvectors that correspond to the ```k``` largest eigenvalues to construct a ```d x k``` -dimensional transformation matrix ```W``` ; the eigenvectors are the columns of this matrix  
        6. project the samples onto the new feature subspace using the transformation matrix ```W```  
- kernel principal component analysis (kernel PCA):  
    + nonlinear dimensionality reduction  
    + transform data that is not linearly separable onto a new, lower-dimensional subspace that is suitable for linear classifiers  
    + kernel PCA: (computationally expensive)
        * perform a nonlinear mapping --> transforms the data onto a higher-dimensional space  
        * use standard PCA in this higher-dimensional space to project the data back onto a lower-dimensional space <-- where the samples can be separated by a linear classifier  
        * downside: have to specify the parameter the priori *__gamma__*  
        * memory based method: have to reuse the original training set each time to project new samples  
    + kernel trick (compute on the original feature space)  
        * compute covariance between two features (standardized feature)  
        ![cov_between_features](https://cloud.githubusercontent.com/assets/5633774/24373096/19aa7afe-12e5-11e7-8627-b16b4b6de210.png)  
        * general equation to calculate the covariance matrix  
        ![covariance matrix](https://cloud.githubusercontent.com/assets/5633774/24373181/62845a4c-12e5-11e7-86aa-83345f492202.png)  
        * new covariance matrix: replace the dot products between samples in the original feature space by the nonlinear feature combinations via **φ**  
        ![kernel trick](https://cloud.githubusercontent.com/assets/5633774/24373251/950484b0-12e5-11e7-8100-2efe11710979.png)  
        the matrix notation of the new covariance matrix:  
        ![kernel_trick_matrix_notation](https://cloud.githubusercontent.com/assets/5633774/24373869/bdb91784-12e7-11e7-82dc-82156490fb31.png)  
        the eigenvector equation:  
        ![eigenvector equation](https://cloud.githubusercontent.com/assets/5633774/24373981/1eb4c768-12e8-11e7-995c-f25e3e45f32a.png)  
        since:  
        ![eigenvector](https://cloud.githubusercontent.com/assets/5633774/24374015/37de54e8-12e8-11e7-9c3e-9a89ecb58549.png)  
        can get:  
        ![image](https://cloud.githubusercontent.com/assets/5633774/24374049/53fbbc4c-12e8-11e7-8610-f2b47ad03099.png)  
        multiply it by φ(X) on both sides:  
        ![image](https://cloud.githubusercontent.com/assets/5633774/24374092/710c3a6e-12e8-11e7-9b67-78c232b9e32b.png)  
        * get the similarity kernel matrix  
        ![similarity kernel matrix](https://cloud.githubusercontent.com/assets/5633774/24373516/6af982a0-12e6-11e7-8221-4b915747d3f7.png)  
    + commonly used kernels:  
        * polynomial kernel:  
        ![polynomial kernel](https://cloud.githubusercontent.com/assets/5633774/24374374/6f9d22f0-12e9-11e7-91e1-5e68b6f6fa8a.png)  
        * hyperbolic tangent kernel (sigmoid):  
        ![sigmoid kernel](https://cloud.githubusercontent.com/assets/5633774/24374391/80ac8572-12e9-11e7-8a54-d064911b3d79.png)  
        * Radial Basis Function (RBF) or Gaussian kernel:  
        ![rbf kernel](https://cloud.githubusercontent.com/assets/5633774/24374408/95539588-12e9-11e7-8717-a508faa58666.png)  
    + RBF kernel PCA steps:  
        1. compute the kernel (similarity) matrix k:  
        ![kernel matrix](https://cloud.githubusercontent.com/assets/5633774/24374778/f6f5777e-12ea-11e7-8e54-6358cb08953f.png)  
        2. center the kernel matrix k --> since cannot guarantee that the new feature space is also centered at zero:  
        ![center kernel matrix](https://cloud.githubusercontent.com/assets/5633774/24374839/2e41403c-12eb-11e7-9d7a-4f885f0a40dc.png)  
        ![1n](https://cloud.githubusercontent.com/assets/5633774/24374956/83bc241e-12eb-11e7-874f-8bc0a940f692.png)  
        note: all values are equal to ```1/n```  
    + project new data points  
        * calculate the pairwise RBF kernel between each i-th sample in the training dataset and the new sample x':  
        ![kernel rbf new data](https://cloud.githubusercontent.com/assets/5633774/24386400/3d4b5a34-1324-11e7-9719-0e39a506cd60.png)  
        * eigenvectors **a** and eigenvalues **λ** of the kernel matrix **K** satisfy the the condition:  
        ![kernel rbf condition](https://cloud.githubusercontent.com/assets/5633774/24386415/53f591dc-1324-11e7-98d7-04b91d1c6f21.png)  
        
        
### assess feature importance  
- use random forest: measure feature importance as the averaged impurity decrease  
    + computed from all decision trees in the forest without making any assumption whether the data is linearly separable or not   
    
    
### logistic regression  
P 454  
- characteristics  
    + linear: performs well on linearly separable classes  
    + binary classification  
    + model for classification not regression  
    + can estimate the class-membership probability  
- odds ratio:  
    + ```p / (1-p)```  
    + p: the probability of the positive event  
- logit function: 
    + the logarithm of the odds ratio (log-odds)  
    + ```logit(p) = log[p / (1-p)]```  
- linear relation between feature values and the log-odds:  
    + ```logit[p(y=1|x)]= w*x```  
    + ```p(y=1|x)```: the conditional probability --> a sample belongs to class 1 given its feature **x**  
- logistic function (sigmoid function)  
    + ![logistic function](https://cloud.githubusercontent.com/assets/5633774/24224530/da415f7e-0f18-11e7-8e12-0d06759edad2.png)  
    + ```z=w*x```   
- convert predicted probability into a binary outcome:
    + ![binary_outcome](https://cloud.githubusercontent.com/assets/5633774/24224793/7274109c-0f1a-11e7-8000-20db892674bc.png)  
- cost function & the cost for the classification for different value of (φ)z:
    ![cost_fun](https://cloud.githubusercontent.com/assets/5633774/24259096/1122f91e-0fad-11e7-84e5-e67f050fb5fb.png)  
    ![cost_fun_single](https://cloud.githubusercontent.com/assets/5633774/24259008/bf5a48bc-0fac-11e7-86e2-b64eb655d56a.png)  
    ![cost_fun_logi](https://cloud.githubusercontent.com/assets/5633774/24258981/a06eb956-0fac-11e7-9e53-f5333d01073c.png)  
- cost function with regularization  
    + λ: L2 regularization parameter --> L2 shrinkage | weight decay  
    ![cost_fun_l2](https://cloud.githubusercontent.com/assets/5633774/24262419/2d388e20-0fb7-11e7-8193-6b926f5c3a8d.png)  
    + C: ```C = 1 / λ``` --> decreasing the value of the inverse regularization parameter C means that we are increasing the regularization strength  
    ![c_l2](https://cloud.githubusercontent.com/assets/5633774/24262636/ec525e58-0fb7-11e7-9055-791fd4d5d4e3.png)  

    
### support vector machines  
P 94  
- SVM: maximum margin classification  
![SVM_margin](https://cloud.githubusercontent.com/assets/5633774/24305291/a7910338-1079-11e7-8abe-777219081137.png)  
    + optimization objective: maximize the margin --> tend to have a lower generalization error (models with small margins are more prone to overfitting)  
    + margin: distance between the separating hyperplane (decision boundary)  
        * positive hyperplane: ![positive_hyperplane](https://cloud.githubusercontent.com/assets/5633774/24305422/1fb16042-107a-11e7-894b-171e16fbd3e6.png)  
        * negative hyperplane: ![negative_hyperplane](https://cloud.githubusercontent.com/assets/5633774/24305436/301ed0e0-107a-11e7-8615-f9eb6085e971.png)  
        * subtract two hyperplanes: ![subtract_hyperplane](https://cloud.githubusercontent.com/assets/5633774/24305521/8019ea76-107a-11e7-9042-78374a27ec3c.png)  
        ![subtract_hyperplane](https://cloud.githubusercontent.com/assets/5633774/24305573/b117943e-107a-11e7-8ced-ad6a35e89962.png)  
        * left side: distance between the positive and negative hyperplane --> the margin to maximize  
        * right side: the objective function of the SVM --> in practice: use quadratic programming ![svm_goal](https://cloud.githubusercontent.com/assets/5633774/24305885/c3ea45b0-107b-11e7-923f-44f7430b7a30.png)  
        * all negative samples should fall on one side of the negative hyperplane; all the positive samples should fall behind the positive hyperplane  
        ![image](https://cloud.githubusercontent.com/assets/5633774/24305751/54238dfe-107b-11e7-8014-624b7e1b8f5e.png)  
    + support vectors: the training samples that are closest to this hyperplane  
- slack variable:
    + soft-margin classification  
    + linear constraints need to be relaxed for nonlinearly separable data: allow convergence of the optimization in the presence of mis-classifications under the appropriate cost penalization  
    ![margin_slack_variable](https://cloud.githubusercontent.com/assets/5633774/24306051/63ee8cb0-107c-11e7-87a0-72ad6fd7ab8b.png)  
    + linear constraints with slack variable:  
    ![linear constraints](https://cloud.githubusercontent.com/assets/5633774/24306178/e1bced80-107c-11e7-91db-3cc55dbc983a.png)
    + objective function with slack variable:  
    ![objective_fun](https://cloud.githubusercontent.com/assets/5633774/24306222/18b46ade-107d-11e7-936c-7498ea739396.png)  
    + C: control the penalty for misclassification --> control the width of the margin --> tune the bias-variance trade-off  
    ![svm_C](https://cloud.githubusercontent.com/assets/5633774/24306426/c514add4-107d-11e7-8f5d-6c200bc4d475.png)  
        * large C: large error penalties  
        * small C: less strict about misclassification errors  
- SVM vs logistic regression  
    + yield similar results  
    + logistic regression: 
        * maximize the conditional likelihoods of the training data --> prone to outliers than SVMs  
        * simpler model  
        * can be easily updated --> streaming data  

        
### kernel SVM  
- SVM can be easily kernelized to solve nonlinear classification  
- create nonlinear combinations of the original features  
    + project original feature onto a higher dimensional space --> on the higher dimensional space: the feature is linearly separable  
    + mapping function: φ(x1, x2) --> could be very expensive  
    ![kernel_projection](https://cloud.githubusercontent.com/assets/5633774/24315749/3f0c4a02-10a6-11e7-89d3-76c30c6df6c9.png)  
        * kernel function:  
        ![kernel_function](https://cloud.githubusercontent.com/assets/5633774/24315920/710b1a96-10a7-11e7-821d-d9fe2013d2e4.png)  
    + widely used kernel:
        * RBF kernel: Radial Basis Function kernel  
        ![rbf](https://cloud.githubusercontent.com/assets/5633774/24315964/bbfd083e-10a7-11e7-9ce8-10e0c8e337a6.png)  
        ![rbf_parameter](https://cloud.githubusercontent.com/assets/5633774/24315976/d28ce6be-10a7-11e7-930c-443e5023a01b.png)  
        * γ (gamma in sklearn: control overfitting): a cut-off parameter for the Gaussian sphere (increase γ will the influence of the training samples --> small γ: softer decision boundary; large γ: tighter boundary --> likely have a high generalization error on unseen data)  
            


### decision tree learning  
P 105  
- decision tree:      
    + break down data by making decisions based on a series of questions --> good interpret-ability  
    + split the data on the feature that results in the largest information gain (IG)  
    + prune: prune the tree by setting a limit for the maximal depth of the tree  
    + boundaries: 
        * can build complex decision boundaries  
        * deeper the decision tree, the more complex the decision boundary become --> result in overfitting  
    + feature scaling is not requirement for decision tree algorithms  
- objective function: maximize the information gain at each split  
![decision_tree_goal](https://cloud.githubusercontent.com/assets/5633774/24317306/123195c0-10b3-11e7-99ad-768abc52dc41.png)  
    + f: the feature to perform the split  
    + I: impurity measure  
    + Dp: the dataset of the parent; Dj: the dataset of the j-th child node  
    + Np: total number of samples at the parent node; Nj: the number of samples in the j-th child node  
- information gain (IG)  
    + split the nodes at the most informative features  
    + the difference between the impurity of the parent node and the sum of the child impurities --> the lower the impurity of the child nodes, the larger the information gain  
- binary decision tree --> reduce the combinatorial search space  
![binary_decision_tree](https://cloud.githubusercontent.com/assets/5633774/24317455/67413664-10b4-11e7-86ca-eaa4001984d2.png)  
- commonly used impurity measures:  
    + Entropy:  
    ![entropy](https://cloud.githubusercontent.com/assets/5633774/24317526/26670230-10b5-11e7-8f74-4f3b96ba04b9.png)  
        * ```p(i|t)```: the proportion of the samples that belongs to class c for a particular node t --> ```entropy=0``` if all samples at a node belong to the same class  
        * the entropy is maximal if there is a uniform class distribution --> all belongs to one class  
        * entropy criterion attempts to maximize the mutual information  
    + Gini index:  
    ![gini](https://cloud.githubusercontent.com/assets/5633774/24317595/db784f44-10b5-11e7-90aa-566855365419.png)  
        * minimize the probability of misclassification  
        * Gini index is maximal if the classes are perfectly mixed      
    + classification error  
    ![classification_error](https://cloud.githubusercontent.com/assets/5633774/24317627/4c5714fc-10b6-11e7-9fe9-dfda3766940b.png)  
        * useful criterion for **pruning**, __not__ for growing a decision tree  

        
### GraphViz program  
P 114  
- open source graph visualization software  
- scikit-learn is allows to export the decision tree as a ```.dot``` file after training, which can be visualized using the GraphViz program  
```
dot -Tpng tree.dot -o tree.png
```


### random forests 
P 115  
- random forests:  
    + ensemble of decision tree: combining weak learners to strong learners  
    + good classification performance, scalability, and ease of use  
    + less hyper-parameters  
        * number of trees: *__k__* --> larger: better performance but increased computational cost  
        * size of bootstrap sample: *__n__* --> control the bias-variance trade off; larger: decrease the randomness==likely to overfitting  
        * number of features: *__d__*: smaller than the total number fo features in the training set (e.g., d=m^0.5)  
    + typically don't need to prune the random forest --> robust to noise  
    
    + four steps:  
        1. draw a random **bootstrap** sample of size *__n__* (randomly choose n samples from the training set with replacement)  
        2. grow a decision tree from the bootstrap sample. At each node:  
            1. randomly select *__d__* features without replacement  
            2. split the node using the feature that provides the best split according to the objective function (e.g., maximize information gain)  
        3. repeat steps 1-2 *__k__* times  
        4. aggregate the prediction by each tree to assign the class label by **majority vote**  
        
        
        
### k-nearest neighbors  
- KNN:  
    ![knn](https://cloud.githubusercontent.com/assets/5633774/24327715/8fdfb5d0-118d-11e7-9f4c-19716f66a0b2.png)  
    + lazy learner: doesn't learn a discriminative function from the training data but memorizes the training dataset instead  
        * classifier immediately adapts as new training data come in  
        * computational complexity grows linearly with the number of samples  
        * cannot discard training samples --> need large storage space for large datasets  
    + very susceptible to overfitting due to the **curse of dimensionality** --> the feature space becomes increasingly sparse for an increasing number of dimensions of a fixed-size training dataset --> feature selection or dimensionality reduction  
    + non-parametric model --> instance-based learning --> lazy learning: no cost during the learning process  
    + three steps:  
        1. choose the number of *__k__* and a distance metric  
        2. find the *__k__* nearest neighbors of the sample that need to classify  
        3. assign the class label by majority vote  
    
    
    
### missing data  
- missing value  
    + blank spaces  
    + placeholder strings: NaN  
- why need to take care of missing value:  
    + most computational tools are unable to handle missing values  
    + computational tools will produce unpredictable results  
- handle missing values:  
    + remove corresponding features (columns) or samples (rows) with missing values  
        * remove too many samples --> make a reliable analysis impossible  
        * lose valuable information --> classifier cannot discriminate between classes  
    + interpolation techniques: imputing missing values  
        * mean imputation: replace the missing value by the mean value of the entire feature column  
        * median imputation: replace the missing value by the median value of the entire feature column  
        * most_frequent imputation: replaces the missing values by the most frequent values  
- handle categorical data:  
    + nominal features  
    + ordinal features: categorical values that can be sorted or ordered  
        * mapping ordinal features: convert the categorical string values into integers  
    + encoding class labels: class labels are required to encoded as integer values  
    + one-hot encoding on nominal features: create a new dummy feature for each unique value in the nominal feature column  
    

    
### feature scaling  
- majority of machine learning and optimization algorithms behave much better if features are on the same scale  
- feature scaling:  
    + normalization: 
        * the rescaling of the features to a range of ```[0,1]``` --> special case of min-max scaling  
        * apply the min-max scaling to each feature column:  
        ![normalization](https://cloud.githubusercontent.com/assets/5633774/24332835/2f7b0ef4-1202-11e7-8c62-eb3c5effcb8d.png)  
    + standardization  
        * center the feature columns at mean ```0``` with standard deviation ```1``` --> feature columns take the form of a normal distribution --> easier for machine learning algorithm to learn the weights  
        * maintains useful information about outliers --> less sensitive to outliers (in contrast to min-max scaling)  
        * apply the standard scaling to each feature column:  
        ![standardization](https://cloud.githubusercontent.com/assets/5633774/24332983/f62d94a2-1204-11e7-9bf4-c9446bab6ecb.png)  
                    
### model selection 
- tuning and comparing different parameter settings to further improve the performance for making predictions on unseen data  



### holdout cross-validation  
P 198  
![model selection](https://cloud.githubusercontent.com/assets/5633774/24388264/6693392a-132e-11e7-935c-1a8e8f02cb70.png)  
- separate the data into three parts: select optimal values --> hyper-parameters:
    + training set: used to fit the different models  
    + validation set: the performance on the validation set is used for the model selection  
    + test set: the model hasn't seen before during the training and model selection steps --> obtain a less biased estimate of its ability to generalize to new data  
- estimate the models' generalization error on the test dataset  
- disadvantage:  
    + sensitive to how to partition the training set into the training and validation subsets  
    + the estimate will vary for different samples of the data  


    
### k-fold cross-validation  
P 200  
![k-fold cross validation](https://cloud.githubusercontent.com/assets/5633774/24388686/3463b350-1331-11e7-8b17-5f2fd7407cec.png)  
- use k-fold cross-validation for model tuning  
    + finding the optimal hyper-parameter values --> yield a satisfying generalization performance  
    + retrain the model on the complete training set and obtain a final performance estimate using the independent test set  
    + steps:  
        * randomly split the training dataset into **_k_** folds without replacement  
        * **_k-1_** folds: model training  
        * **_1_** fold: testing  
        * repeated k times --> obtain **_k_** model  
- tuning k:  
    + small training sets --> larger k --> useful to increase the number of folds --> more training data will be used in each iteration, which results in a lower bias towards estimating the generalization performance by averaging the individual model estimates  
    + large dataset --> small k --> still obtain an accurate estimate of the average performance of the model while reducing the computational cost of refitting and evaluating the model on the different folds  
- leave-one-out (LOO): 
    + set the number of folds equal to the number of training samples (k = n), only one training sample is used for testing during each iteration   
    + for working with very small datasets  
- stratified k-fold cross-validation  
    + yield better bias and variance estimates (especially in cases of unequal class proportions)  
    + class proportions are preserved in each fold --> ensure each fold is representative of the class proportions in the training dataset  
    
    
    
### learning curves & validation curves  
P 204  
- learning curves  
    + diagnose if a learning algorithm has a problem with overfitting (high variance) or underfitting (high bias)  
    ![learning curves](https://cloud.githubusercontent.com/assets/5633774/24414375/1e42e778-1393-11e7-88cd-d6faf37e2aa9.png)  
        * high bias: both low training and cross-validation accuracy --> underfits the training data --> increase the number of parameters  
        * high variance: large gap between the training and cross-validation accuracy --> collect more training data; reduce the complexity of the model  
- validation curves  
    + improve the performance of a model by addressing the common issues such as overfitting or underfitting  
    
 
 
### grid search  
P 210  
- fine-tuning parameters of machine learning models  
- brute-force exhaustive search paradigm  



### nested cross-validation  
P 187  
![nested cross-validation](https://cloud.githubusercontent.com/assets/5633774/24419332/a0023d94-13a3-11e7-9001-19733f5653f1.png)  
- algorithm selection  
- steps:  
    1. an outer k-fold cross-validation loop to split the data into training and test folds  
    2. an inner loop used to select the model using k-fold cross-validation on the training fold  
    3. test fold is used to evaluate the model performance  
    
    
### performance evaluation metrics  
P 214  
- confusion matrix: square matrix that reports the counts of the true positive, true negative, false positive, and false negative predictions of a classifier  
![confusion matrix](https://cloud.githubusercontent.com/assets/5633774/24420041/40738b28-13a6-11e7-8746-3e4f735d1cee.png)  
- **error** (**ERR**) and **accuracy** (**ACC**)  
    + provide general information about how many samples are misclassified  
    + ERR: the sum of all false predictions divided by the number of total predications  
    ![err](https://cloud.githubusercontent.com/assets/5633774/24420501/e9ffbb16-13a7-11e7-8696-5a68b4efa97b.png)  
    + ACC: accuracy is calculated as the sum of correct predictions divided by the total number of predictions  
    ![acc](https://cloud.githubusercontent.com/assets/5633774/24420528/ff71faa4-13a7-11e7-8506-ff9e208c8f72.png)  
- **true positive rate** (**TPR**) and **false positive rate** (**FPR**)  
    + useful for imbalanced class problems  
    ![fpr tpr](https://cloud.githubusercontent.com/assets/5633774/24420628/59eb6e7a-13a8-11e7-8e3a-080b10848949.png)  
- **precision** (**PRE**) and **recall** (**REC**)  
    ![pre rec](https://cloud.githubusercontent.com/assets/5633774/24420849/18d2586c-13a9-11e7-9c58-46fbcb30f6a5.png)  
- **F1-score**  
    + combination of precision and recall  
    ![f1](https://cloud.githubusercontent.com/assets/5633774/24420897/4650418c-13a9-11e7-9019-926db831dcc7.png)  
- **receiver operator characteristic** (**ROC**) and **area under the curve** (**AUC**)  
    + useful tools for selecting models for classification  based on their performance with respect to the false positive and true positive rates  
    + computed by shifting the decision threshold of the classifier  
    + perfect classifier would fall into the top-left corner of the graph with a true positive rate of 1 and a false positive rate of 0  
    + AUC is calculated based on ROC  
- scoring metrics for multi-class classification  
    + micro-average: calculated from the **individual** true positives, true negatives, false positives, and false negatives of the system  
    ![micro pre](https://cloud.githubusercontent.com/assets/5633774/24430557/fb579eaa-13cb-11e7-9855-94b183b387d6.png)  
    + macro average: calculated as the **average scores** of the different systems (useful with class imbalances)  
    ![macro pre](https://cloud.githubusercontent.com/assets/5633774/24430575/0c776df0-13cc-11e7-91b7-40b1aa03f1a6.png)  

    
    
### ensembles  
P 224  
- combine different classifiers tin to a meta-classifier that has a better generalization performance than each individual classifier alone  
- the error probability of an ensemble is always better than the error of an individual base classifier as long as the base classifiers perform better than random guessing (50%)  
- ensemble learning increases the computational complexity compared to individual classifiers  


### majority voting (binary class) & plurality voting (multi-class):  
    ![majority voting](https://cloud.githubusercontent.com/assets/5633774/24431232/92c86d66-13cf-11e7-90cf-16b07bdd347a.png)  
    + select the class label that has been predicted by the majority of classifiers  
    ![majority voting mode](https://cloud.githubusercontent.com/assets/5633774/24431255/b6b937be-13cf-11e7-975f-e0d981f03ce2.png)  
    + two ways:
        1. training m different classifiers (C1...Cm) on training set  
        2. use the same base classification algorithm fit different subsets fo the training set  
    + majority voting (weighted): combine different classification algorithms associated with individual weights for confidence --> build a stronger meta-classifier that balances out the individual classifiers' weaknesses on a particular dataset  
    ![weighted majority voting](https://cloud.githubusercontent.com/assets/5633774/24517503/772e2612-1533-11e7-9cda-106677a296f1.png)  
        * **_w_**: a weight associated with a base classifier  
        * **_C_**: the predicted class label of the ensemble  
        * **_χ_**: the characteristic function  
        * **_A_**: the set of unique class labels  
    + majority voting (equal weights):  
    ![equal weight](https://cloud.githubusercontent.com/assets/5633774/24517844/a901d2c8-1534-11e7-8d8e-cf9878d46cc6.png)  
    + majority voting (probabilities): using the predicted class probabilities instead of the class labels for majority voting --> useful if the classifiers in the ensemble are well calibrated  
    ![majority voting prob](https://cloud.githubusercontent.com/assets/5633774/24518208/0639f8de-1536-11e7-868c-fb6662abac50.png)  
        + **_Pij_**: the predicted probability of the j-th classifier for class label i  



### bagging
P 244  
- building an ensemble of classifiers from bootstrap samples  
- complex classification (data with high dimensionality) can easily lead to overfitting, bagging is:  
    * effectively reduce the **variance of a model**  
    * ineffective in reducing model bias (in practice, choose ensemble of classifiers with low bias, e.g., unpruned decision trees)  
- bootstrap: random samples with replacement  
- bootstrap aggregating:  
    + draw bootstrap samples from the initial training set in each round of bagging  
    + each bootstrap sample is used to fit a classifier (e.g., an unpruned decision tree. random forests are a special case of bagging)  
    
    
### adaptive boosting  
P 249  
- leveraging weak learners  
    + the ensemble consists of very simple base classifiers (weak learners)  
    + let the weak learners subsequently learn from misclassified training samples to improve the performance of the ensemble  
- can lead to decrease in bias as well as variance compared to bagging models (in practice: lead to high variance <-- tendency to overfit the training data)  
- boosting steps:
    1. draw a random subset of training samples ```d1``` without replacement from the training set ```D``` to train a weak learner ```C1```  
        * all training samples are assigned equal weights  
    2. draw second random training subset ```d1``` without replacement from the training set and add 50% of the samples that were previously misclassified to train a weak learner ```C2```  
        * assign a larger weight to the previously misclassified samples  
        * lower the weight of the correctly classified samples  
    3. find the training samples ```d3``` in the training set ```D``` on which ```C1``` and ```C2``` disagree to train a third weak learner ```C3```  
    4. combine the weak learners ```C1``` , ```C2``` , and ```C3```  via majority voting  
- AdaBoost steps:  
![adaboost](https://cloud.githubusercontent.com/assets/5633774/24531494/627a19b6-156e-11e7-9c75-f6a69f8a0f88.png)  


### sentiment analysis  
P 260  
- sentiment analysis == opinion mining  
- analyze the polarity of documents: classify documents based on the expressed opinions or emotions of the authors with regard to a particular topic  
     
     
     
### bag-of-words model  
P 261  
- represent text as numerical feature vectors  
    + create a vocabulary of unique tokens from the entire set of documents  
    + construct a feature vector from each document that contains the counts of how often each word occurs in the particular document  
- unigram model: each item or token in the vocabulary represents a single word  
- n-gram: the contiguous sequences of items in NLP  
    + choice of the number n in the n-gram model depends on the particular application  
    
    
### term frequency inverse document frequency (tf-idf)  
P 263  
- down-weight the frequently occurring words in the feature vectors  
![tf-idf](https://cloud.githubusercontent.com/assets/5633774/24536217/cbed5948-158d-11e7-9535-2986ac484f56.png)  
![idf](https://cloud.githubusercontent.com/assets/5633774/24536229/dc90b75e-158d-11e7-9df1-37331a996d8e.png)  
- add one to idf:
![tf-idf+1](https://cloud.githubusercontent.com/assets/5633774/24536512/bdb77186-158f-11e7-9748-98f1c28501f2.png)  
![idf+1](https://cloud.githubusercontent.com/assets/5633774/24536526/d931d550-158f-11e7-9272-7bfdd239e0c2.png)  
    + to avoid division by zero, as when a term appears in no documents, even though this would not happen in a strictly "bag of words" approach  
    + to set a lower bound to avoid a term being given a zero weight just because it appeared in all documents  
- normalize the raw term frequencies: L2-normalization --> returns a vector of length 1 by dividing an unnormalized feature vector v by its L2-norm  
![l2-norm](https://cloud.githubusercontent.com/assets/5633774/24536559/0be7957a-1590-11e7-8185-7d31b8521727.png)  
  

  
### regression analysis  
P 302  
- predict target variables on a continuous scale  
    + simple (univariate) linear regression: model the relationship between a single feature (explanatory variable **_x_**) and a continuous valued response (target variable **_y_**)  
    ![regression model](https://cloud.githubusercontent.com/assets/5633774/24570038/33f9c28e-161e-11e7-9172-08cd05b5c36c.png)  
        * goal: learn the weights of the linear equation to describe the relationship between the explanatory variable and the target variable, which can then be used to predict the responses of new explanatory variables that were not part of the training dataset  
        * regression line: best-fitting line  
        ![regression line](https://cloud.githubusercontent.com/assets/5633774/24570084/83aa4290-161e-11e7-92f5-b81cfd319638.png)  
    + multiple linear regression:  
    ![multiple linear regression](https://cloud.githubusercontent.com/assets/5633774/24570113/ab5e94da-161e-11e7-86e5-44b2a67263e4.png)  
    + polynomial regression: model a nonlinear relationship  
    ![polynomial regressiion](https://cloud.githubusercontent.com/assets/5633774/24573104/d147b590-1633-11e7-8328-38f022ff6b80.png)  
- ordinary least squares (**OLS**): estimate the parameters of the regression line that minimizes the sum of the squared vertical distances (residuals or errors) to the sample points  
![ols](https://cloud.githubusercontent.com/assets/5633774/24571298/0be01ea4-1625-11e7-8bed-d4e7938938ad.png)  
- residual plots: 
    + goal:  
        * graphical analysis for diagnosing regression models to detect nonlinearity and outliers  
        * check if the errors are randomly distributed  
    + perfect prediction: residuals will be exactly zero  
    + good model: errors are randomly distributed and the residuals should be randomly scattered around the centerline    
- Mean Squared Error (**MSE**): comparing different regression models or for tuning their parameters via a grid search and cross-validation  
![mse](https://cloud.githubusercontent.com/assets/5633774/24572480/e5150826-162d-11e7-89d6-b0e3bbb16c4a.png)  
- coefficient of determination (**R^2**): standardized version of the MSE [0, 1]  
    + the fraction of response variance that is captured by the model   
    ![R^2](https://cloud.githubusercontent.com/assets/5633774/24572583/a74cfa2a-162e-11e7-8810-123fecb09bfc.png)  
    SSE: the sum of squared errors  
    SST: the total sum of squares  
    ![sst](https://cloud.githubusercontent.com/assets/5633774/24572590/b8c99e70-162e-11e7-916f-16a71ecf4897.png)  
    + if R^2=1, the model fits the data perfectly with a corresponding MSE=0  
- regularization:  
    + Ridge Regression: L2 penalized model --> add the squared sum of the weights to the least-squares cost function  
    By increasing the value of the hyperparameter λ --> increase the regularization strength and shrink the weights of our model  
    ![ridge regression](https://cloud.githubusercontent.com/assets/5633774/24572668/a9ca94aa-162f-11e7-8ddc-9bbc97d30bb9.png)  
    ![l2](https://cloud.githubusercontent.com/assets/5633774/24572674/bef16048-162f-11e7-9e52-f817025d2486.png)  
    + Least Absolute Shrinkage and Selection Operator (LASSO): depending on the regularization strength, certain weights can become zero, which makes the LASSO also useful as a supervised feature selection technique  
    limitation: selects at most ```n``` variables if ```m > n```  
    ![lasso](https://cloud.githubusercontent.com/assets/5633774/24572709/0d368986-1630-11e7-92fb-c607a47e6256.png)  
    ![l1](https://cloud.githubusercontent.com/assets/5633774/24572715/2118eeee-1630-11e7-95c7-db56d8c39ba4.png)  
    + Elastic Net: has a L1 penalty to generate sparsity and a L2 penalty to overcome some of the limitations of the LASSO  
    ![elastic net](https://cloud.githubusercontent.com/assets/5633774/24572748/60113232-1630-11e7-9ece-ce20c0dd73ae.png)    


    
### decision tree regression  
P 329  
- does not require any transformation of the features wit nonlinear data  
- replace entropy as the impurity measure of a node by the MSE (within-node variance) --> variance reduction  
![tree_mse](https://cloud.githubusercontent.com/assets/5633774/24573346/f157aa4a-1636-11e7-867f-0c98663b35d3.png)  
- predict target value  
![tree_pre](https://cloud.githubusercontent.com/assets/5633774/24573367/16bb8716-1637-11e7-8eb3-1c22015a6836.png)  
  
    
    
### random forest regression 
P 331  
- less sensitive to outliers in the dataset and don't require much parameter tuning  
- use the MSE criterion to grow the individual decision trees --> the predicted target variable is calculated as the average prediction over all 
decision trees  


    
    
### RANdom SAmple Consensus (RANSAC)  
- ovoid the impact by the presence of outliers  
- steps:  
    1. Select a random number of samples to be inliers and fit the model  
    2. Test all other data points against the fitted model and add those points that fall within a user-given tolerance to the inliers  
    3. Refit the model using all inliers  
    4. Estimate the error of the fitted model versus the inliers  
    5. Terminate the algorithm if the performance meets a certain user-defined threshold or if a fixed number of iterations has been reached; go back to step 1 otherwise  
    

   
    
### exploratory data analysis  
P 305  
- visually detect the presence of outliers, the distribution of the data, and the relationships between features  
- first step prior to the training of a machine learning model  
- scatter plot matrix:  
    + visualize the pair-wise correlations between the different features in this dataset in one place  
    + how the data is distributed and whether it contains outliers  
- correlation matrix:  
    + the linear relationship between the features  
    + a square matrix that contains the Pearson product-moment correlation  coefficients ```[-1, 1]``` --> measure the linear dependence between pairs of features  
    ![pearson correlation](https://cloud.githubusercontent.com/assets/5633774/24571046/bc2ec438-1623-11e7-8f90-fec17a109cbf.png)

    
    
### clustering
P 366  
- goal: find a natural grouping in data such that items in the same cluster are more similar to each other than those from different clusters --> group the samples based on their feature similarities  
- hard clustering: each sample in a dataset is assigned to exactly one cluster  
- soft clustering (fuzzy clustering): assign a sample to one or more clusters (e.g., fuzzy C-means (FCM))  


### k-means  
P 337  
- goal: group the samples based on their feature similarities  
    + advantage: easy to implement & computationally efficient  
    + disadvantage: have to specify the number of clusters k a priori  
        * result in bad clusterings  
        * result in slow convergence if the initial centroids are chosen poorly  
    + disadvantage: bad performance if clusters are overlap or not hierarchical  
- optimization: minimize the within-cluster sum of squared errors (SSE)  
![SSE](https://cloud.githubusercontent.com/assets/5633774/24575310/e7e8bcfe-1656-11e7-80d4-0ab16f4763ef.png)  
- steps:  
    1. randomly pick ```k``` centroids from the sample points as initial cluster centers  
    2. assign each sample to the nearest centroid 
    ![knn](https://cloud.githubusercontent.com/assets/5633774/24575279/24b01034-1656-11e7-9a96-2e7e8861d4d2.png)  
    3. move the centroids to the center of the samples that were assigned to it  
    4. repeat the steps 2 and 3 until the cluster assignment do not change or a user-defined tolerance or a maximum number of iterations is reached  
- measure similarity between objects: distance  
    + squared Euclidean distance:  
    ![squared euclidean distance](https://cloud.githubusercontent.com/assets/5633774/24575298/ae0497e2-1656-11e7-8da0-ebe632e5b7e5.png)  
- elbow method --> find the optimal number of clusters  
    + if ```k``` increase, within-cluster SSE will decrease: because the samples will be closer to the centroids they are assigned to  
    + identify the value of ```k``` where the distortion begins to increase most rapidly  
- silhouette plots --> quantify the quality of clustering  
    + a graphical tool to plot a measure of how tightly grouped the samples in the clusters are  
    + scrutinize the sizes of the different clusters and identify clusters that contain outliers  
    + steps:  
        1. calculate the cluster cohesion ```a_i``` as the average distance between a sample ```x_i``` and all other points in the same cluster  
        2. calculate the cluster separation ```b_i``` from the next closest cluster as the average distance between the sample ```x_i``` and all samples in the nearest cluster  
        3. calculate the silhouette ```s_i``` as the difference between cluster cohesion and separation divided by the greater of the two, as shown here:  
        ![silhouette](https://cloud.githubusercontent.com/assets/5633774/24581096/8b1d5c28-16c9-11e7-8c76-97015c4d88c0.png)  
    
    
    
    
### fuzzy C-means (FCM)  
P 343  
- replace the hard cluster assignment by probabilities for each point belonging to each cluster  
    + each iteration is more expensive than an iteration in k-means  
    + requires **fewer** iterations overall to reach convergence  
- objective function:  
![fcm-objective fun](https://cloud.githubusercontent.com/assets/5633774/24575447/a492b1ae-165a-11e7-9c0a-c82a85e11f59.png)  
    + the larger the value of ```m```, the smaller the cluster membership ```w``` becomes, which leads to fuzzier clusters  
    + the cluster membership probability:  
    ![cluster membership probability](https://cloud.githubusercontent.com/assets/5633774/24575461/17fb0b64-165b-11e7-9192-458adaa6a256.png)  
- steps:  
![fcm steps](https://cloud.githubusercontent.com/assets/5633774/24575452/ba41bc48-165a-11e7-8a37-a61fccc24217.png)  

    
    
### k-means++
P 340  
- address k-means issue: place the initial centroids far away from each other --> leads to better and more consistent results  
- steps:  
![k-means++](https://cloud.githubusercontent.com/assets/5633774/24575340/f994dd6a-1657-11e7-8f76-f34731874a87.png)  


### hierarchical clustering  
P 351  
- allows users to plot dendrograms --> help with the interpretation of the results by creating meaningful taxonomies  
    +  do not need to specify the number of clusters upfront  
- dendrograms: visualizations of a binary hierarchical clustering  
- main approaches:  
    + agglomerative hierarchical: start with each sample as an individual cluster and merge the closest pairs of clusters until only one cluster remains  
        * steps (complete linkage):  
            1. compute the distance matrix of all samples  
            2. represent each data point as a singleton cluster  
            3. merge the two closest clusters based on the distance of the most dissimilar (distant) members  
            4. update the similarity matrix  
            5. repeat steps 2 to 4 until one single cluster remains  
        ![single and complete linkage](https://cloud.githubusercontent.com/assets/5633774/24582063/5c124368-16dc-11e7-862b-6494bcdbaaad.png)  
        * single linkage: compute the distances between the most similar members for each pair of clusters and merge the two clusters for which the distance between the most similar members is the smallest  
        * complete linkage: compare the most dissimilar members to perform the merge  
        * average linkage: merge the cluster pairs based on the minimum average distances between all group members in the two clusters  
        * Ward's linkage: two clusters that lead to the minimum increase of the total within-cluster SSE are merged  
    + divisive hierarchical: start with one cluster that encompasses all our samples; iteratively split the cluster into smaller clusters until each cluster only contains one sample  