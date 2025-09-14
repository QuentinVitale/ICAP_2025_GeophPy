from ..visualization.decorators import grid_plot_setup
from ..visualization import helpers
import matplotlib.pyplot as plt

class MagneticPlottingMixin:
    """Mixin for magnetism-specific plotting methods."""

    def plot_euler_solutions(self, solutions, fig=None, ax=None, **kwargs):
        """
        Overlays Euler deconvolution solutions on a grid map.

        This is a high-level plotting method that first draws the survey's
        surface map and then overlays the calculated Euler solutions as a
        colored scatter plot.

        Parameters
        ----------
        solutions : pandas.DataFrame
            The DataFrame of solutions returned by the euler_deconvolution method.
        fig : matplotlib.figure.Figure, optional
            An existing figure to plot on.
        ax : matplotlib.axes.Axes, optional
            An existing axes to plot on. If None, a new figure/axes are created.
        **kwargs
            Additional keyword arguments for the background surface plot
            (e.g., `cmap`, `vmin`, `vmax`).
        """
        # --- 1. Handle Figure and Axes Setup ---
        if ax is None:
            if fig is None:
                fig, ax = plt.subplots(figsize=(8, 8))
            else:
                fig.clf()
                ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        # --- 2. Draw the Background Surface Map ---
        # We call the low-level surface plot function directly
        from ..visualization import plot2D
        mappable = plot2D.plot_surface(self, fig=fig, ax=ax, **kwargs)

        # --- 3. Overlay the Euler Solutions ---
        # The size and color of the points can represent depth
        sc = ax.scatter(solutions['x0'], solutions['y0'],
                        s=50, c=solutions['z0_depth'],
                        cmap='hot_r', edgecolor='k',
                        vmin=kwargs.get('vmin_euler'), vmax=kwargs.get('vmax_euler'))
        
        # --- 4. Add a Specific Colorbar for the Solutions ---
        helpers.create_colorbar(fig, ax, sc, cbar_label='Estimated Depth', **kwargs)
        
        # The helper function will handle the title from kwargs
        helpers.apply_plot_options(self, fig, ax, **kwargs)

        return fig, ax