# Python Machine Learning  

reading notes of [this book](https://www.amazon.com/dp/B00YSILNL0/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1) 

# Reading notes 


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
    + for unsupervised data compression  
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
- linear discriminant analysis (**LDA**): supervised dimensionality reduction technique for maximizing class separability  
- kernel principal component analysis: nonlinear dimensionality reduction  
        

    
### assess feature importance  
    + use random forest: measure feature importance as the averaged impurity decrease --> computed from all decision trees in the forest without making any assumption whether the data is linearly separable or not   
    
    
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
                    


    
        