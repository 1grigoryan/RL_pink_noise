import numpy as np

class OrnsteinUhlenbeckNoise:
    def __init__(self, size, theta=0.15, sigma=0.2, dt=1e-2, seed=None):
        """OU noise for continuous action spaces.

        Parameters
        ----------
        size : int or tuple of int
            Shape of the noise to be generated.
        theta : float
            The rate at which the noise decays to its mean.
        sigma : float
            The scale of the noise.
        dt : float
            The time step.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.seed = seed
        self.reset()

    def reset(self):
        self.state = np.zeros(self.size)
        if self.seed is not None:
            np.random.seed(self.seed)

    def sample(self):
        noise = np.random.normal(size=self.size)
        self.state += self.theta * (-self.state) * self.dt + self.sigma * np.sqrt(self.dt) * noise
        return self.state