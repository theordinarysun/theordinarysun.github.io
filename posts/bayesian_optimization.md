Write-up for Bayesian Optimization, Roman Garnett.


# Introduction to Gaussian Process
An auxiliary step in optimization is modeling the objective function $f: \mathcal{X} \rightarrow \mathbb{R}$. The typical probabilistic approach is to define distributions over its observations, which are treated as random variables due to their inherent randomness. When $\mathcal{X}$ is infinite, the function modeled by this approach is regarded as a \textit{stochastic process}. A stochastic process that comprises of multivariate Gaussian distributions is thus called a \textit{Gaussian process}.

When $\mathcal{X}$ is finite, a Gaussian process is equivalent to a multivariate normal distribution. Yet, the thing that sets a Gaussian process aside from a multivariate normal distribution is its use of kernel function in place of covariance matrix to accommodate any arbitrary $x$ (in the infinite $\mathcal{X}$). Indeed, a Gaussian process on $f$ is fully specified by its mean function $\mu: \mathcal{X} \rightarrow \mathbb{R}$ and kernel function $\mat{K}: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ ($\mat{K} \succcurlyeq 0$):
$$p(f) = \mathcal{GP}(f; \mu, \mat{K})$$

The marginal distribution of finite observations $\boldsymbol{\phi} = f(\vect{x})$ is thus defined as
$$
p(\boldsymbol{\phi} \mid f, \vect{x}) = \mathcal{N}(\boldsymbol{\phi}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) \qquad \text{where} \qquad \boldsymbol{\mu} = \mu(\vect{x}); \quad \boldsymbol{\Sigma} = \mat{K}(\vect{x}, \vect{x})
$$

# Bayesian Inference on Gaussian Process
Doing Bayesian inference on a Gaussian process means updating this probabilistic model with new observations using Bayes' rule. Inference is crucial because it provides the framework for a Gaussian process to learn from data.

Recall that the Bayes' rule states $p(f | \boldsymbol{\phi}) = \frac{p(\boldsymbol{\phi} | f, \vect{x}) p(f)}{p(\boldsymbol{\phi} | \vect{x})}$ where we have known from above $p(f) = \mathcal{GP}(f; \mu, \mat{K})$ and $p(\boldsymbol{\phi}|f, \vect{x}) = \mathcal{N}(\boldsymbol{\phi}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$. The missing ingredient here is $p(\boldsymbol{\phi} | \vect{x})$. It would be ideal if the posterior function is another Gaussian process. A choice of $p(\boldsymbol{\phi} | \vect{x})$ that makes it feasible is a multivariate Gaussian distribution.
$$
p(\boldsymbol{\phi} \mid \vect{x}) = \mathcal{N}(\boldsymbol{\phi} | \mathbf{m}, \mathbf{C})
$$
The posterior function becomes
$$
p(f \mid \boldsymbol{\phi}) = \mathcal{GP} 
\left( 
\bmat{f \\ \boldsymbol{\phi}}; \bmat{\mu \\ \mathbf{m}}, \bmat{\mat{K} & \kappa^T \\ \kappa & \mat{C}}
\right)
$$
where $\kappa$ is the cross-covariance matrix between $f$ and $\boldsymbol{\phi}$.


# Modeling with Gaussian Process
Effective modeling of the objective function is crucial for optimization in general.
In Bayesian optimization, the observations or function values are modeled by two components, namely the prior model and the observation model, intervene by the Bayes' rule.
    - The observation model: An observation $y$ given an input $x$ is often modeled as a noisy output of $f$ perturbed by Gaussian noise with mean $0$ and variance $\sigma^2$: $$ y = f(x) + \xi \qquad \text{where} \qquad \xi \sim \mathcal{N}(x; 0, \sigma^2) $$
    - The prior model: A common choice of prior is to specify $f$ as a Gaussian process, which is fully specified by its prior mean and kernel function. By following common practice, a good chance you will end up with a constant mean function and a squared exponential kernel function.
    $$
    \mu(x) = c; \qquad \mat{K}(x, x') = \exp \left( \frac{-|x-x'|^2}{2} \right)
    $$
    There are actually good reasons for that, and you will see in a moment when we will spend the rest of this section talking about these moments.

## The Prior Mean Function

## The Prior Kernel Function