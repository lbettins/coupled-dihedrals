
import numpy as np

def augment_with_periodic_bc(points, values, domain):
    """
    Augment the data to create periodic boundary conditions.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    domain : float or None or array_like of shape (n, )
        The size of the domain along each of the n dimenions
        or a uniform domain size along all dimensions if a 
        scalar. Using None specifies aperiodic boundary conditions.

    Returns
    -------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions with 
        periodic boundary conditions.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions with periodic
        boundary conditions.
    """
    # Validate the domain argument
    n = len(points)
    if np.ndim(domain) == 0:
        domain = [domain] * n
    if np.shape(domain) != (n,):
        raise ValueError("`domain` must be a scalar or have the same "
                         "length as `points`")

    # Pre- and append repeated points
    points = [x if d is None else np.concatenate([x - d, x, x + d]) 
              for x, d in zip(points, domain)]

    # Tile the values as necessary
    reps = [1 if d is None else 3 for d in domain]
    values = np.tile(values, reps)

    return points, values
