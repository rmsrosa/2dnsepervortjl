@def title = "The pressure-velocity formulation"

# {{ get_title }}

The equations, in the vectorial Eulerian formulation for the velocity field $\mathbf{u}$ and the kinematic pressure $p$, take the form

$$
\begin{cases}
  \displaystyle \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \boldsymbol{\nabla})\mathbf{u} + \boldsymbol{\nabla} p = \nu \Delta \mathbf{u} + \mathbf{f}, \\
  \boldsymbol{\nabla} \cdot \mathbf{u} = 0.
\end{cases}
$$

These are assumed to hold on a two-dimensional spatial domain $\Omega = (0, L)^2$, where $L>0$, and on a time interval $t \in [t_0, T]$, where we tipically assume $t_0 = 0$.

The forcing term $\mathbf{f}$ is assumed to be time-independent, square-integrable over $\Omega$, and with zero average over $\Omega$.

We are further given a zero-average, square-integrable, and divergence-free initial condition $\mathbf{u}(t_0) = \mathbf{u_0}$.

Under these conditions, there is a unique solution $\mathbf{u}$, which has zero average over $\Omega$ and its $L$-periodic extension is locally in $H^1$, at almost every time $t \geq t_0$.
