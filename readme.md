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
    




