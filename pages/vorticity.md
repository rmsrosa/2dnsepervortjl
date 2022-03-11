@def title = "Vorticity formulation"

# {{ get_title }}

An alternative formulation is with the vorticity. In the two-dimensional case, the vorticity is (essentially) a scalar, and the pressure disappears, making it easier to solve the system.

When the velocity field is $\mathbf{u} = (u, v)$ and the spatial variable is $\mathbf{x} = (x, y)$, the (scalar) vorticity is given by
$$ \omega = v_x - u_y,
$$
where the subscripts denote the corresponding partial derivatives.

Alternatively, if $\mathbf{u} = (u, v, 0)$, then $\mathbf{\omega} = \boldsymbol{\nabla} \times \mathbf{u} = (0, 0, \omega)$.

By taking the curl of the linear momentum equation, we obtain

$$
\partial_t (\boldsymbol{\nabla} \times \mathbf{u}) + \boldsymbol{\nabla} \times ((\mathbf{u} \cdot \boldsymbol{\nabla} \mathbf{u})\cdot \mathbf{u}) + \boldsymbol{\nabla} \times \boldsymbol{\nabla} p = \nu \Delta (\boldsymbol{\nabla} \times \mathbf{u}) + \boldsymbol{\nabla} \times \mathbf{f}.
$$

Notice that
$$
\begin{align*}
  \displaystyle \boldsymbol{\nabla} \times \boldsymbol{\nabla} p & = (0, 0, 0), \\
  \displaystyle \partial_t (\boldsymbol{\nabla} \times \mathbf{u}) & = (0, 0, \omega_t), \\
  \displaystyle \nu \Delta (\boldsymbol{\nabla} \times \mathbf{u}) & = (0, 0, \nu\Delta\omega), \\
  \displaystyle \boldsymbol{\nabla} \times \mathbf{f} & = (0, 0, g),
 \end{align*}
$$
where $g = (f_1)_y - (f_2)_x$, where $f_1$ and $f_2$ are the two compenents of the force, $\mathbf{f} = (f_1, f_2)$.

As for the remaining term, we write 
$$
\boldsymbol{\nabla} \times ((\mathbf{u}\cdot\boldsymbol{\nabla})\mathbf{u}) = \boldsymbol{\nabla} \times \left( \begin{matrix} uu_x + vu_y \\ uv_x + vv_y \\ 0 \end{matrix}  \right)
$$
Considering only the third component since the others vanish, we find
$$
\begin{align*}
(uv_x + vv_y)_x - (uu_x + vu_y)_y & = u_xv_x + uv_{xx} + v_xv_y + vv_{yx} - u_yu_x - uu_{xy} - v_yu_y  -vu_{yy} \\
  & = u(v_{xx} -u_{xy}) + v(v_{yx}-u_{yy}) + u_x(v_x-u_y) + v_y(v_x-u_y) \\
  & = u\omega_x + v\omega_y + u_x\omega + v_y\omega = (u\omega)_x + (v\omega)y = \boldsymbol{\nabla}\cdot(\omega\mathbf{u}).
\end{align*}
$$

Combining the expressions and considereing only the $z$-component, we find the two-dimensional vorticity equation
$$
  \omega_t + \boldsymbol{\nabla}\cdot(\omega\mathbf{u}) = \nu\Delta \omega + g.
$$

Due to the divergence-free condition on the velocity field, we have
$$
\boldsymbol{\nabla}\cdot(\omega\mathbf{u}) = \boldsymbol{\nabla}\omega \cdot \mathbf{u} + \omega \boldsymbol{\nabla}\cdot \mathbf{u} = \boldsymbol{\nabla}\omega \cdot \mathbf{u}.
$$

But we keep the form $\boldsymbol{\nabla}\cdot(\omega\mathbf{u}),$ in order to apply the Basdevant reduction, which requires less Fourier transforms; see [{{get_title /pages/basdevant}}](/pages/basdevand).
