import numpy as np

def gabor_filter(
    X,
    Y,
    # # Parameters (replace with extracted values from your data)
    # A=1.0,  # Amplitude
    theta=np.pi / 4,  # Preferred orientation (radians)
    f=0.5,  # Spatial frequency
    sigma=1.0,  # Gaussian envelope width along x
    gamma=1.0,
    phi_0=np.pi / 3,  # Preferred phase (radians)
    theta_gauss=None,
    **kwargs
):

    # Rotate coordinates
    x_prime = X * np.cos(theta) + Y * np.sin(theta)
    y_prime = -X * np.sin(theta) + Y * np.cos(theta)

    if theta_gauss:
        x_prime_gauss = x_prime * np.cos(theta_gauss) + y_prime * np.sin(theta_gauss)
        y_prime_gauss = -x_prime * np.sin(theta_gauss) + y_prime * np.cos(theta_gauss)
    else:
        x_prime_gauss = x_prime
        y_prime_gauss = y_prime

    # Gabor function
    return (
        np.exp(
            -(
                x_prime_gauss**2 / (2 * sigma**2)
                + gamma**2 * y_prime_gauss**2 / (2 * sigma**2)
            )
        )
        * np.cos(2 * np.pi * f * x_prime + phi_0)
    )


def gabor_rate_response(X, Y, params, img, mode="simple"):

    if mode == "simple":
        G = gabor_filter(X, Y, **params)
        print(G.mean(), np.abs(G).mean())
        print(img.mean(), np.abs(img).mean())

        G -= G.mean()
        img -= img.mean()

        ## construct proper einsum arguments
        string = ""
        for i in range(len(img.shape) - 2):
            string += chr(ord("a") + i)
        einsum_string = f"ij,{string}ij->{string}"

        return np.einsum(einsum_string, G, img, order="C")
    elif mode == "complex":
        rate_even = gabor_rate_response(X, Y, params, img, mode="simple")
        rate_odd = gabor_rate_response(X, Y, params | {"phi_0": params["phi_0"] + np.pi / 2}, img, mode="simple")
        return np.sqrt(rate_even**2 + rate_odd**2)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def elliptical_pdf(X, Y, x0=0.0, y0=0.0, sigma_x=1.0, sigma_y=1.0, angle=0.0, amplitude=None, return_log=False, eps=1e-12):
    """
    Evaluate a 2D elliptical Gaussian PDF at coordinates (X, Y).

    Parameters
    - X, Y: array-like of identical shape (can be meshgrid arrays)
    - x0, y0: center position of the ellipse
    - sigma_x, sigma_y: standard deviations along the ellipse principal axes (>0)
    - angle: rotation of the principal axes in radians (counterclockwise)
    - amplitude: if None, uses normalized PDF amplitude 1/(2*pi*sigma_x*sigma_y).
                    If provided, that value is used as a multiplicative prefactor.
    - return_log: if True, return the log-PDF instead of PDF values
    - eps: small floor for sigmas to avoid division by zero

    Returns
    - array of same shape as X/Y with PDF (or log-PDF) values
    """
    sigma_x = max(float(sigma_x), eps)
    sigma_y = max(float(sigma_y), eps)

    # Shift coordinates to center
    dx = X - x0
    dy = Y - y0

    # Precompute sin/cos
    c = np.cos(angle)
    s = np.sin(angle)

    # Components of the inverse covariance matrix Σ^{-1} for rotated Gaussian
    inv_sx2 = 1.0 / (sigma_x * sigma_x)
    inv_sy2 = 1.0 / (sigma_y * sigma_y)

    a = c * c * inv_sx2 + s * s * inv_sy2
    b = s * c * (inv_sx2 - inv_sy2)  # off-diagonal term (will be used as 2*b*dx*dy)
    c_comp = s * s * inv_sx2 + c * c * inv_sy2  # named c_comp to avoid shadowing

    exponent = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c_comp * dy * dy)

    if amplitude is None:
        norm = 1.0 / (2.0 * np.pi * sigma_x * sigma_y)
    else:
        norm = float(amplitude)

    if return_log:
        return np.log(norm) + exponent
    else:
        return norm * np.exp(exponent)


def sine_grating(X, Y, theta, f, phi_0, square=False):
    # Rotate coordinates
    x_prime = X * np.cos(theta) + Y * np.sin(theta)

    # Gabor function
    if square:
        return np.sign(np.cos(2 * np.pi * f * x_prime + phi_0))
    else:
        return np.cos(2 * np.pi * f * x_prime + phi_0)


def softplus(x, alpha=1., gamma=0., delta = 0.):
    # return alpha * np.log(1 + np.exp((x - gamma) / alpha)) + delta
    return alpha * np.log(1 + np.exp((x - gamma) / alpha)) + delta


def ReLU(x, alpha, beta, theta):
    return alpha * np.maximum(beta, x - theta)


# def gompertz_from_specs(M, slope_at_inflect, x_at_small, eps_small):
#     """
#     Returns a Gompertz y(x) with:
#       - Max = M
#       - Slope s at inflection
#       - y(x_at_small) = eps_small (tiny baseline)
#     """
#     k = np.e * slope_at_inflect / M
#     x0 = x_at_small - (1.0 / k) * np.log(np.log(M / eps_small))
#     def y(x):
#         return M * np.exp(-np.exp(-k * (x - x0)))
#     return y, dict(M=M, k=k, x0=x0)

# def gompertz(x, M, k, x0):
#     return M * np.exp(-np.exp(-k * (x - x0)))
