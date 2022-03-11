@def title = "The pressure-velocity formulation"

# {{ get_title }}

The equations, in the vectorial Eulerian formulation for the velocity field $\mathbf{u}$ and the kinematic pressure $p$, take the form

$$
\begin{cases}
  \displaystyle \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \boldsymbol{\nabla} \mathbf{u})\cdot \mathbf{u} + \boldsymbol{\nabla} p = \nu \Delta \mathbf{u} + \mathbf{f}, \\
  \boldsymbol{\nabla} \cdot \mathbf{u} = 0.
\end{cases}
$$

These are assumed to hold on a two-dimensional domain $\Omega = (0, L)^2$, where $L>0$, and on a time interval $t \in [t_0, T]$.

We are further given a initial condition $\mathbf{u}(t_0) = \mathbf{u_0}$, where we tipically assume $t_0 = 0$. And we assume that $\mathbf{u}$ is $L$-periodic in both directions, in the sense that its $L$-periodic extension is a "smooth" function, i.e. belonging at least to $H_{\textrm{loc}}^1(\mathbb{R}^2)$.

The forcing term $\mathbf{f}$ is assumed to be $L$-periodic as well, time-independent, and with zero average over $\Omega$.
