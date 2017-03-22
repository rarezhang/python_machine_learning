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
    
