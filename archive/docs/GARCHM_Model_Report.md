
# GARCH-M Model Report

## 1. Model Introduction

The GARCH-M (Generalized Autoregressive Conditional Heteroskedasticity-in-Mean) model extends the traditional GARCH framework by incorporating conditional volatility directly into the mean equation. This is particularly relevant in financial applications, where asset returns may depend on their own volatility due to risk-return tradeoffs. In addition to the GARCH structure, our model includes observable factors that may influence asset returns.

### Mean Equation
The GARCH-M model assumes the following form for the conditional mean of returns:

\[ R_t = \alpha + \beta^\top F_t + \lambda \cdot \sqrt{h_t} + \varepsilon_t \]

- \( R_t \): Asset return at time \( t \)
- \( \alpha \): Intercept term
- \( \beta \): Vector of factor loadings
- \( F_t \): Vector of observable factors at time \( t \)
- \( \lambda \): Coefficient for the conditional standard deviation (volatility-in-mean effect)
- \( h_t \): Conditional variance
- \( \varepsilon_t \): Zero-mean error term

### Variance Equation
The conditional variance \( h_t \) follows a GARCH(p, q) process:

\[ h_t = \omega + \sum_{i=1}^{p} \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^{q} \beta_j h_{t-j} \]

- \( \omega \): Constant term
- \( \alpha_i \): ARCH parameters (response to past squared residuals)
- \( \beta_j \): GARCH parameters (response to past variances)



## 2. Estimation Method

The parameters of the GARCH-M model are estimated by maximizing the log-likelihood function under the assumption of normally distributed errors:

\[ \mathcal{L}(\theta) = -\frac{1}{2} \sum_t \left[ \log(h_t) + \frac{\varepsilon_t^2}{h_t} \right] \]

The parameter vector \( \theta \) includes:

\[ \theta = (\alpha, \beta, \lambda, \omega, \alpha_1, \ldots, \alpha_p, \beta_1, \ldots, \beta_q) \]

Numerical optimization (e.g., SLSQP) is used to estimate \( \theta \), with constraints imposed to ensure positivity and stationarity of the variance process:

- \( \omega > 0 \), \( \alpha_i \geq 0 \), \( \beta_j \geq 0 \)
- \( \sum \alpha_i + \sum \beta_j < 1 \) (for covariance stationarity)



## 3. Significance Testing


After estimation, the Hessian matrix of the log-likelihood is approximated numerically. Standard errors are computed from the inverse of the Hessian:

\[ \text{Var}(\hat{\theta}) \approx H^{-1} \quad \Rightarrow \quad \text{SE}(\hat{\theta}_i) = \sqrt{(H^{-1})_{ii}} \]

Then, the t-statistics and p-values are calculated as:

\[ t_i = \frac{\hat{\theta}_i}{\text{SE}(\hat{\theta}_i)}, \quad p_i = 2(1 - \Phi(|t_i|)) \]

However, we find that the calculated t-statistics are not valid as the standard errors are not robust. As we are not able to improve the standard errors, we cannot use them to conduct robust hypothesis tests. This is a limitation of our current approach.


## 4. Implementation

To implement the GARCH-M model, we should determine the orders of the GARCH process (p, q). We simply apply GARCH model on the return rates to determine the orders as the GARCH-M model is a special case of the GARCH model which may have similar autocorrelation structures. We use the AIC and BIC criteria to select the optimal orders. The selected orders are then used in the GARCH-M model.
We also filter the factors to select a subset of factors to include in the model. As we mentioned in factor modeling, we have a large number of factors which may lead to overfitting and make the calculation costly. To avoid this, we conduct a preliminary analysis to select a subset of factors.
Then we fit the GARCH-M model using the selected factors and orders. The model is estimated using maximum likelihood estimation (MLE) with numerical optimization methods. 
For each stock, we use a period of 500 days to fit the model and then predict the return rate of next day. And we apply the model to all stocks using rolling window approach. The model is re-estimated every day.
