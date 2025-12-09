# Collaborative Filtering : Matrix Factorization

Implemented collaborative filtering to make recommendations to users U for items I. Used a model based approach that applies matrix factorization to factorize a rating matrix (𝑅) into User features (𝑃) and Item features (𝑄). Solution is implemented using gradient descent technique to perform matrix factorization. Given a rating matrix, the code applies matrix factorization to determine User features (P), Item features (Q), User bias vector (bu) and Item bias vector (bu).

Following operations are performed:
- Computing gradients: Applying gradient descent to minimize mean square error. Computing gradient of error with respect to p and q values and deriving formulas to update p and q values.
- Adding biases: There can be biases in user and item recommendations, which are handled by introducing user bias vector (bU) and item bias vector (bI). In the presence of these biases, rating 𝑟𝑖𝑗 is predicted.
