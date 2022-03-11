@def title = "Basdevant reduction"

# {{ get_title }}

In this way, we need to apply the inverse fft to three functions, namely on $\hat\omega$, $\hat u$, $\hat v$, and then two direct fft, namely on $\omega u$ and $\omega v$. Alternatively, we can keep it as $\boldsymbol{\nabla}\omega \cdot \mathbf{u}$ and compute four ifft and one fft.

The best, however, is to perform the Basdevant reduction to reduce to four ffts (two inverse and two direct). This is obtained by writing
$$
\boldsymbol{\nabla} \cdot (\omega\mathbf{u}) = \left(\partial_x^2 - \partial_y^2\right)(uv) + \partial_{xy}\left(v^2 - u^2\right).
$$
This way, we need to compute the inverse fft of $u$ and $v$, compute the product $uv$ and the term $v^2 - u^2$ in physical space, and finally compute the FFT of these two functions.