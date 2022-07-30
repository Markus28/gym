"""A wrapper that adds human-renering functionality to an environment."""
# from threading import Thread
from multiprocessing import Manager, Process
from queue import Queue

import numpy as np

import gym
from gym.error import DependencyNotInstalled


def render_frames_sync(q, size, fps):
    try:
        import pygame
    except ImportError:
        raise DependencyNotInstalled(
            "pygame is not installed, run `pip install gym[box2d]`"
        )
    pygame.init()
    pygame.display.init()
    window = pygame.display.set_mode(size[::-1])
    clock = pygame.time.Clock()

    while True:
        frame = q.get()

        if frame is None:
            print("Finishing")
            pygame.display.quit()
            pygame.quit()
            print("Done")
            return

        frame = np.transpose(frame, axes=(1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        window.blit(surf, (0, 0))
        pygame.event.pump()
        clock.tick(fps)
        pygame.display.flip()


class HumanRendering(gym.Wrapper):
    """Performs human rendering for an environment that only supports rgb_array rendering.

    This wrapper is particularly useful when you have implemented an environment that can produce
    RGB images but haven't implemented any code to render the images to the screen.
    If you want to use this wrapper with your environments, remember to specify ``"render_fps"``
    in the metadata of your environment.

    The ``render_mode`` of the wrapped environment must be either ``'rgb_array'`` or ``'single_rgb_array'``.

    Example:
        >>> env = gym.make("LunarLander-v2", render_mode="single_rgb_array")
        >>> wrapped = HumanRendering(env)
        >>> wrapped.reset()     # This will start rendering to the screen

    The wrapper can also be applied directly when the environment is instantiated, simply by passing
    ``render_mode="human"`` to ``make``. The wrapper will only be applied if the environment does not
    implement human-rendering natively (i.e. ``render_mode`` does not contain ``"human"``).

    Example:
        >>> env = gym.make("NoNativeRendering-v2", render_mode="human")      # NoNativeRendering-v0 doesn't implement human-rendering natively
        >>> env.reset()     # This will start rendering to the screen

    Warning: If the base environment uses ``render_mode="rgb_array"``, its (i.e. the *base environment's*) render method
        will always return an empty list:

            >>> env = gym.make("LunarLander-v2", render_mode="rgb_array")
            >>> wrapped = HumanRendering(env)
            >>> wrapped.reset()
            >>> env.render()
            []          # env.render() will always return an empty list!

    """

    def __init__(self, env):
        """Initialize a :class:`HumanRendering` instance.

        Args:
            env: The environment that is being wrapped
        """
        super().__init__(env, new_step_api=True)
        self._is_open = True
        assert env.render_mode in [
            "single_rgb_array",
            "rgb_array",
        ], f"Expected env.render_mode to be one of 'rgb_array' or 'single_rgb_array' but got '{env.render_mode}'"
        assert (
            "render_fps" in env.metadata
        ), "The base environment must specify 'render_fps' to be used with the HumanRendering wrapper"

        self.screen_size = None
        self.manager = Manager()
        self.frame_queue = self.manager.Queue(maxsize=5)
        self.t = None

    @property
    def render_mode(self):
        """Always returns ``'human'``."""
        return "human"

    def step(self, *args, **kwargs):
        """Perform a step in the base environment and render a frame to the screen."""
        result = self.env.step(*args, **kwargs)
        self._render_frame()
        return result

    def reset(self, *args, **kwargs):
        """Reset the base environment and render a frame to the screen."""
        result = self.env.reset(*args, **kwargs)
        self._render_frame()
        return result

    def render(self):
        """This method doesn't do much, actual rendering is performed in :meth:`step` and :meth:`reset`."""
        return None

    def _render_frame(self, mode="human", **kwargs):
        """Fetch the last frame from the base environment and render it to the screen."""
        if self.env.render_mode == "rgb_array":
            last_rgb_array = self.env.render(**kwargs)
            assert isinstance(last_rgb_array, list)
            last_rgb_array = last_rgb_array[-1]
        elif self.env.render_mode == "single_rgb_array":
            last_rgb_array = self.env.render(**kwargs)
        else:
            raise Exception(
                f"Wrapped environment must have mode 'rgb_array' or 'single_rgb_array', actual render mode: {self.env.render_mode}"
            )
        assert isinstance(last_rgb_array, np.ndarray)

        if mode == "human":
            if self.t is None:
                # self.t = Thread(target=render_frames_sync, args=(self.frame_queue, last_rgb_array.shape[:-1], self.env.metadata["render_fps"]))
                self.t = Process(
                    target=render_frames_sync,
                    args=(
                        self.frame_queue,
                        last_rgb_array.shape[:-1],
                        self.env.metadata["render_fps"],
                    ),
                )
                self.t.start()

            self.frame_queue.put(last_rgb_array)
        else:
            raise Exception("Can only use 'human' rendering in HumanRendering wrapper")

    def close(self):
        """Close the rendering window."""
        self._is_open = False
        # super().close()
        self.frame_queue.put(None)
        if self.t is not None and self.t.is_alive():
            self.frame_queue.put(None)
            self.t.join()
        super().close()

    def __del__(self):
        if self._is_open:
            self.close()
