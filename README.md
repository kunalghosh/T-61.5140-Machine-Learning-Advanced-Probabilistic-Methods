# T-61.5140-Machine-Learning-Advanced-Probabilistic-Methods
Project work done as a part of the course on Machine Learning Advanced Probabilistic Methods held in Aalto University during Spring 2016

In this project work we had the following tasks:
1. To write down the mathematical description for a mixture model with two linear components:
    * Full posterior likelihood function.
    * Full log posterior likelihood function.
2. To derive the EM update equations for the parameters of this model:
    * Update equations for φ1 and φ2
    * Update equations for σ_21 and σ_22
    * Update equation for w
    * Update equation for zt
3. To implement the above model:
    * Model initialization
    * Log posterior likelihood function
    * Update equations
4. We had to then test the model:
    * Generate training and validation data from the mixture model. Analyze how well the model (trained with the training data) can explain the validation data with different data dimensionality and different amounts of generated data when you only do the fitting once. How do the results change, and why, when you start the EM from multiple locations and choose the best fit? 
 
5. Finally we had to compare the two models (simple linear model and mixture with two linear components) and do the analyses with both low (eg. 2) and high (eg. 10) data dimensionality as well as with small (eg. 10) and large (eg. 100) amount of samples. We used separate validation set as before. Some additional tasks were:
    * Draw data from the simple linear model, analyze how well each of the candidate models is able to explain the data.
    * Draw data from the mixture model, analyze how well each of the candidate models is able to explain the data. 
    * Draw data from the mixture model, analyze which candidate model is able to explain the data better as a function of the similarity of the two linear components in the true model (e.g. cosine similarity). We finally had to explain our findings.
