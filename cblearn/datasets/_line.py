from cblearn.datasets._base import BaseManifold
import numpy as np
from sklearn.utils import check_random_state
from scipy.spatial.distance import pdist, squareform
from typing import Union, Dict, Callable

class Line(BaseManifold):
    """
    Line

    Line is a class for creating a random line in a higher dimensional space.
    It provides methods to sample points from the line and calculate distances
    between those points.

    Attributes:
        space_dimension: Dimensionality of the space
        random_state: Random state for reproducibility of the line

    Examples:
        >>> from your_module import Line
        >>> line = Line(space_dimension=3, random_state=42)
        >>> points = line.sample_points(num_points=10)
        >>> distances = line.get_canonical_distance_matrix(points)
        >>> print(points.shape)
        (10, 3)
        >>> print(distances.shape)
        (10, 10)
    """
    def __init__(self, space_dimension: int, random_state: Union[None, int, np.random.RandomState] = None):
        """
        Initialize the Line

        Args:
            space_dimension: Dimensionality of the space
            random_state: The seed of the pseudo random number generator
                          to use when sampling. If None, the random number
                          generator is the RandomState instance used by
                          np.random.
        """
        if not isinstance(space_dimension, int):
            raise ValueError('Space dimension must be an integer')
        if space_dimension < 1:
            raise ValueError('Space dimension cannot be less than 1')
        self.space_dimension = space_dimension
        self.random_state = check_random_state(random_state)
        self.origin = None
        self.direction = None
        super().__init__(subspace_dimension=1,
                         space_dimension=space_dimension,
                         random_state=self.random_state)

    def _create_manifold(self):
        """Create the random line"""
        self.origin = self.random_state.rand(self.space_dimension)
        self.direction = self.random_state.randn(self.space_dimension)  # Random direction vector
        self.direction /= np.linalg.norm(self.direction)  # Normalize direction vector

    def sample_points(self, num_points: int,
                      sampling_function: Union[str, Callable] = 'normal',
                      sampling_options: Dict = {'scale': 1},
                      noise: Union[None, str, Callable] = None,
                      noise_options: Dict = {},
                      random_state: Union[None, int, np.random.RandomState] = None,
                      return_distances: bool = True) -> np.ndarray:
        """
        Sample points from the line

        Args:
            num_points: Number of points to sample
            sampling_function: The sampling function to use.
                               If a string, it should be a method of
                               the random state object. If a callable,
                               it should be a function that takes a
                               size argument and returns a numpy array
                               of samples.
            sampling_options: The options to pass to the sampling function.
            noise: The noise function to use. If a string, it should be
                   a method of the random state object. If a callable,
                   it should be a function that takes a size argument and
                   returns a numpy array of samples.
            noise_options: The options to pass to the noise function.
            random_state: The seed of the pseudo random number generator
                          to use when sampling. If None, the random number
                          generator is the RandomState instance used by
                          np.random.
            return_distances: Flag to return the distance matrix of
                              the sampled points. Defaults to True.

        Returns:
            Sampled points as a numpy array of shape (num_points, space_dimension).
            If return_distances is True, the distance matrix of the sampled points
            is also returned.
        """
        if self.origin is None or self.direction is None:
            self._create_manifold()
        
        # Set random_state
        if random_state is not None:
            random_state = check_random_state(random_state)

        # Get Sampling Function
        if isinstance(sampling_function, str):
            sampling_fun: Callable = getattr(self.random_state, sampling_function)
        elif callable(sampling_function):
            sampling_fun = sampling_function
        else:
            sampling_fun = None
        
        if sampling_fun is None:
            t_values = self.random_state.rand(num_points)
        else:
            t_values = sampling_fun(size=num_points, **sampling_options)
        
        # Vectorized calculation of points
        points = self.origin + np.expand_dims(t_values, axis=-1) * self.direction

        # Add noise if requested
        if noise is not None:
            if isinstance(noise, str):
                noise_fun: Callable = getattr(self.random_state, noise)
            elif callable(noise):
                noise_fun = noise
            else:
                noise_fun = None
            
            if noise_fun is not None:
                noise = noise_fun(size=points.shape, **noise_options)
                points = points + noise

        if return_distances:
            return points, self.get_canonical_distance_matrix(points)
        else:
            return points

    def get_canonical_distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate the pairwise distances between points

        Args:
            points: Array of points for which distances are to be calculated

        Returns:
            Distance matrix of the points (num_points, num_points)
        """
        return squareform(pdist(points))
