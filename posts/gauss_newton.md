# Gauss-Newton approximation of Hessian

## Notations
Let $f_w: \mathbb{R}^d \rightarrow \mathbb{R}^m$ denote the mapping from input space of $d$-dimension to output space of $m$-dimension.
The operator $\sigma$ denotes the non-linear activation on the output of $f$.
We call $z = f_w(x)$ logits and $\sigma(z)$ activation outputs (e.g. softmax outputs if $\sigma = \frac{\exp(z)}{\sum_{i=0}^{m} \exp{(z)}}$).
We find the set of parameters $w$ for $f$ by minimising the loss function $\mathcal{L}$.
Ultimately, we want to find the Hessian matrix $H = \nabla^2_w \mathcal{L}(\sigma(z))$.
In this post, we will look at Gauss-Newton matrix $G$ that approximates Hessian exactly at optimum for realisable problems (our algorithms can solve perfectly).
$$G = J^\intercal_{zw} H_{z} J_{zw}$$
where $J = \frac{\partial z}{\partial w} = \frac{\partial}{\partial w} f_w(x)$ is the Jacobian matrix of raw outputs (logits) and $H_z = \frac{\partial^2 \mathcal{L}}{\partial z^2}$.

## Derivation
We start from our definition of $H$. Particularly, element $(i, j)$ in $H$ is defined as
$$H_{ij} = \frac{\partial}{w_i} \frac{\partial}{w_j} \mathcal{L}(\sigma(z))$$
Use chain rule ($\frac{d}{dx} f(u(x)) = \frac{df}{du} \frac{du}{dx}$) to expand $\frac{\partial}{\partial w_j} \mathcal{L}(\sigma(z))$, we get
$$
H_{ij} 
= \frac{\partial}{w_i}\left[ \sum_{k=0}^m \frac{\partial \mathcal{L}_\sigma}{\partial z_k} \frac{\partial z_k}{\partial w_j}\right] 
= \sum_{k=0}^m \frac{\partial}{w_i}\left[ \frac{\partial \mathcal{L}_\sigma}{\partial z_k} \frac{\partial z_k}{\partial w_j}\right]$$
Apply product rule ($\frac{d}{dx}(uv) = u\frac{dv}{dx} + v\frac{du}{dx}$)
$$
\frac{\partial}{w_i} \left[ \frac{\partial \mathcal{L}_\sigma}{\partial z_k} \frac{\partial z_k}{\partial w_j} \right]
= \frac{\partial \mathcal{L}_\sigma}{\partial z_k} \frac{\partial^2 z_k}{\partial w_i w_j} + \frac{\partial z_k}{\partial w_j} \underbrace{\frac{\partial}{\partial w_i}\left(\frac{\partial \mathcal{L}_\sigma}{\partial z_k}\right)}_{(a)}
$$
Now let's deal with $(a)$ using the same chain rule as before
$$\frac{\partial}{\partial w_i}\left( \frac{\partial \mathcal{L}_\sigma}{\partial z_k} \right)
= \frac{\partial}{\partial z_k} \left( \frac{\partial \mathcal{L}_\sigma}{w_i} \right)
= \frac{\partial}{\partial z_k} \sum_{l=0}^m \frac{\partial \mathcal{L}_\sigma}{\partial z_l} \frac{\partial z_l}{\partial w_i}
= \sum_{l=0}^m \frac{\partial^2 \mathcal{L}_\sigma}{\partial z_k \partial z_l} \frac{\partial z_l}{\partial w_i} 
$$
Plugging everything back, we arrive at
$$H_{ij}
= \sum_{k=0}^m \frac{\partial \mathcal{L}_\sigma}{\partial z_k} \frac{\partial^2 z_k}{\partial w_i w_j} + \sum_{k=0}^m \sum_{l=0}^m \frac{\partial z_k}{w_j} \frac{\partial^2 \mathcal{L}_\sigma}{\partial z_k \partial z_l} \frac{\partial z_l}{\partial w_i}
$$
Equivalently,
$$H = \sum_{k=0}^{m} J_{z_k} (H_{zw})_k + J_{zw}^\intercal H_z J_{zw}$$