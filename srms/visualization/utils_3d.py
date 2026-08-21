"""High-quality 3D visualization of fields on sphere and torus for research papers.

Renders the prediction or ground-truth fields on their actual 3D surfaces with obstacles,
start/goal markers, paths, and optional contours. Both static (matplotlib) and interactive
(Plotly) versions available with multi-angle views.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None
import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


@dataclasses.dataclass
class Sphere3DView:
    """Render a scalar field on the unit sphere in 3D."""

    resolution: int = 100
    """Grid resolution for the sphere mesh."""

    figsize: tuple[int, int] = (10, 8)
    """Figure dimensions in inches."""

    cmap: str = "viridis"
    """Colormap for the field values."""

    elev: float = 20.0
    """Elevation angle for the camera (degrees)."""

    azim: float = 45.0
    """Azimuth angle for the camera (degrees)."""

    vmin: float | None = None
    """Minimum value for color scale (auto if None)."""

    vmax: float | None = None
    """Maximum value for color scale (auto if None)."""

    def render(
        self,
        field: np.ndarray | jnp.ndarray,
        start: tuple[float, float, float] | None = None,
        goal: tuple[float, float, float] | None = None,
        obstacles: list[tuple[float, float, float, float]] | None = None,
        path: np.ndarray | None = None,
        title: str = "Sphere Field",
        out_path: str | None = None,
    ) -> None:
        """Render a field on the sphere.

        Args:
            field: [H, W] array of scalar values on the sphere (lon/lat grid).
            start: (x, y, z) start position on sphere; marked with red star.
            goal: (x, y, z) goal position on sphere; marked with green star.
            obstacles: List of (lon, lat, radius, height_offset) for obstacle markers.
            path: [N, 3] array of path points in 3D Cartesian coordinates.
            title: Title for the plot.
            out_path: Save to file if provided; show() if None.
        """
        field = np.asarray(field)
        if field.ndim != 2:
            raise ValueError(f"field must be 2D, got shape {field.shape}")

        height, width = field.shape
        lon = np.linspace(-np.pi, np.pi, width)
        lat = np.linspace(-np.pi / 2, np.pi / 2, height)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Convert lon/lat to 3D Cartesian on unit sphere
        cos_lat = np.cos(lat_grid)
        x = cos_lat * np.cos(lon_grid)
        y = cos_lat * np.sin(lon_grid)
        z = np.sin(lat_grid)

        # Prepare field for coloring (handle NaN for obstacles)
        field_clean = np.where(np.isnan(field), np.nanmax(field), field)
        vmin = self.vmin if self.vmin is not None else float(np.nanmin(field_clean))
        vmax = self.vmax if self.vmax is not None else float(np.nanmax(field_clean))

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Plot surface with field values
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        ax.plot_surface(x, y, z, facecolors=plt.get_cmap(self.cmap)(norm(field_clean)), shade=False, rstride=1, cstride=1)

        # Add contours on the surface (for reference)
        ax.contour(x, y, z, levels=5, colors="white", alpha=0.3, linewidths=0.5)

        if path is not None:
            path = np.asarray(path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color="cyan", linewidth=2.5, alpha=0.9, zorder=4, label="Path")

        if start is not None:
            ax.scatter(*start, color="red", s=200, marker="*", edgecolors="white", linewidths=1, zorder=5, label="Start")

        if goal is not None:
            ax.scatter(*goal, color="lime", s=200, marker="*", edgecolors="white", linewidths=1, zorder=5, label="Goal")

        if obstacles:
            for lon_obs, lat_obs, rad_obs, height_off in obstacles:
                cos_lat_obs = np.cos(lat_obs)
                x_obs = (1.0 + height_off) * cos_lat_obs * np.cos(lon_obs)
                y_obs = (1.0 + height_off) * cos_lat_obs * np.sin(lon_obs)
                z_obs = (1.0 + height_off) * np.sin(lat_obs)
                ax.scatter(x_obs, y_obs, z_obs, color="orange", s=150, marker="o", edgecolors="red", linewidths=2, zorder=4)

        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_zlabel("Z", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.set_box_aspect([1, 1, 1])

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.6, label="Field Value")

        if start is not None or goal is not None or path is not None:
            ax.legend(loc="upper left", fontsize=9)

        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved sphere visualization to {out_path}")
        else:
            plt.show()
        plt.close(fig)

    def render_multiangle(
        self,
        field: np.ndarray | jnp.ndarray,
        start: tuple[float, float, float] | None = None,
        goal: tuple[float, float, float] | None = None,
        obstacles: list[tuple[float, float, float, float]] | None = None,
        path: np.ndarray | None = None,
        title: str = "Sphere Field",
        out_dir: str = "figures",
    ) -> None:
        """Render the field from multiple viewpoints for paper figures.

        Creates 4 views: front, back, top, isometric.

        Args:
            field: [H, W] array of scalar values.
            start: (x, y, z) start position.
            goal: (x, y, z) goal position.
            obstacles: List of obstacle descriptors.
            path: [N, 3] path points.
            title: Base title for plots.
            out_dir: Directory to save figures.
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        angles = [
            (30, 0, "front"),
            (30, 180, "back"),
            (90, 0, "top"),
            (20, 45, "isometric"),
        ]

        for elev, azim, angle_name in angles:
            self.elev, self.azim = elev, azim
            out_path = f"{out_dir}/{title.lower().replace(' ', '_')}_{angle_name}.png"
            self.render(field, start=start, goal=goal, obstacles=obstacles, path=path, title=f"{title} ({angle_name})", out_path=out_path)


@dataclasses.dataclass
class Torus3DView:
    """Render a scalar field on a flat torus in 3D (embedded in R^3)."""

    major_radius: float = 1.0
    """Major radius of the torus."""

    minor_radius: float = 0.3
    """Minor radius of the torus."""

    resolution: int = 100
    """Grid resolution for the torus mesh."""

    figsize: tuple[int, int] = (10, 8)
    """Figure dimensions in inches."""

    cmap: str = "viridis"
    """Colormap for the field values."""

    elev: float = 20.0
    """Elevation angle for the camera (degrees)."""

    azim: float = 45.0
    """Azimuth angle for the camera (degrees)."""

    vmin: float | None = None
    """Minimum value for color scale (auto if None)."""

    vmax: float | None = None
    """Maximum value for color scale (auto if None)."""

    def render(
        self,
        field: np.ndarray | jnp.ndarray,
        start: tuple[float, float] | None = None,
        goal: tuple[float, float] | None = None,
        obstacles: list[tuple[float, float, float]] | None = None,
        path: np.ndarray | None = None,
        title: str = "Torus Field",
        out_path: str | None = None,
    ) -> None:
        """Render a field on the torus.

        Args:
            field: [H, W] array of scalar values on the torus (θ1, θ2 grid).
            start: (θ1, θ2) start position in angle space.
            goal: (θ1, θ2) goal position in angle space.
            obstacles: List of (θ1, θ2, radius) obstacle centers and radii in angle space.
            path: [N, 2] array of path points in angle space, converted to 3D.
            title: Title for the plot.
            out_path: Save to file if provided; show() if None.
        """
        field = np.asarray(field)
        if field.ndim != 2:
            raise ValueError(f"field must be 2D, got shape {field.shape}")

        height, width = field.shape
        theta1 = np.linspace(-np.pi, np.pi, width)
        theta2 = np.linspace(-np.pi, np.pi, height)
        theta1_grid, theta2_grid = np.meshgrid(theta1, theta2)

        # Convert to 3D Cartesian embedding of torus: R = major, r = minor
        R, r = self.major_radius, self.minor_radius
        x = (R + r * np.cos(theta2_grid)) * np.cos(theta1_grid)
        y = (R + r * np.cos(theta2_grid)) * np.sin(theta1_grid)
        z = r * np.sin(theta2_grid)

        # Prepare field for coloring
        field_clean = np.where(np.isnan(field), np.nanmax(field), field)
        vmin = self.vmin if self.vmin is not None else float(np.nanmin(field_clean))
        vmax = self.vmax if self.vmax is not None else float(np.nanmax(field_clean))

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Plot surface with field values
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=plt.get_cmap(self.cmap)(norm(field_clean)),
            shade=False,
            rstride=1,
            cstride=1,
        )

        # Add contours on the surface
        stride = max(1, height // 5)
        ax.contour(x[::stride], y[::stride], z[::stride], levels=4, colors="white", alpha=0.3, linewidths=0.5)

        if path is not None:
            path = np.asarray(path)
            path_3d = np.zeros((len(path), 3))
            for i, (t1, t2) in enumerate(path):
                path_3d[i, 0] = (R + r * np.cos(t2)) * np.cos(t1)
                path_3d[i, 1] = (R + r * np.cos(t2)) * np.sin(t1)
                path_3d[i, 2] = r * np.sin(t2)
            ax.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2], color="cyan", linewidth=2.5, alpha=0.9, zorder=4, label="Path")

        if start is not None:
            t1_start, t2_start = start
            x_start = (R + r * np.cos(t2_start)) * np.cos(t1_start)
            y_start = (R + r * np.cos(t2_start)) * np.sin(t1_start)
            z_start = r * np.sin(t2_start)
            ax.scatter(x_start, y_start, z_start, color="red", s=200, marker="*", edgecolors="white", linewidths=1, zorder=5, label="Start")

        if goal is not None:
            t1_goal, t2_goal = goal
            x_goal = (R + r * np.cos(t2_goal)) * np.cos(t1_goal)
            y_goal = (R + r * np.cos(t2_goal)) * np.sin(t1_goal)
            z_goal = r * np.sin(t2_goal)
            ax.scatter(x_goal, y_goal, z_goal, color="lime", s=200, marker="*", edgecolors="white", linewidths=1, zorder=5, label="Goal")

        if obstacles:
            for t1_obs, t2_obs, rad_obs in obstacles:
                x_obs = (R + r * np.cos(t2_obs)) * np.cos(t1_obs)
                y_obs = (R + r * np.cos(t2_obs)) * np.sin(t1_obs)
                z_obs = r * np.sin(t2_obs)
                ax.scatter(x_obs, y_obs, z_obs, color="orange", s=120, marker="o", edgecolors="red", linewidths=2, zorder=4)

        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_zlabel("Z", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.set_box_aspect([1, 1, 1])

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.6, label="Field Value")

        if start is not None or goal is not None or path is not None:
            ax.legend(loc="upper left", fontsize=9)

        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved torus visualization to {out_path}")
        else:
            plt.show()
        plt.close(fig)

    def render_multiangle(
        self,
        field: np.ndarray | jnp.ndarray,
        start: tuple[float, float] | None = None,
        goal: tuple[float, float] | None = None,
        obstacles: list[tuple[float, float, float]] | None = None,
        path: np.ndarray | None = None,
        title: str = "Torus Field",
        out_dir: str = "figures",
    ) -> None:
        """Render the field from multiple viewpoints for paper figures.

        Creates 4 views: front, back, top, isometric.

        Args:
            field: [H, W] array of scalar values.
            start: (θ1, θ2) start position.
            goal: (θ1, θ2) goal position.
            obstacles: List of obstacle descriptors.
            path: [N, 2] path points.
            title: Base title for plots.
            out_dir: Directory to save figures.
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        angles = [
            (30, 0, "front"),
            (30, 180, "back"),
            (90, 0, "top"),
            (20, 45, "isometric"),
        ]

        for elev, azim, angle_name in angles:
            self.elev, self.azim = elev, azim
            out_path = f"{out_dir}/{title.lower().replace(' ', '_')}_{angle_name}.png"
            self.render(
                field,
                start=start,
                goal=goal,
                obstacles=obstacles,
                path=path,
                title=f"{title} ({angle_name})",
                out_path=out_path,
            )


def render_sphere_comparison(
    ground_truth: np.ndarray | jnp.ndarray,
    prediction: np.ndarray | jnp.ndarray,
    error: np.ndarray | jnp.ndarray | None = None,
    start: tuple[float, float, float] | None = None,
    out_dir: str = "figures",
) -> None:
    """Create a three-panel comparison of ground-truth, prediction, and error on the sphere.

    Args:
        ground_truth: [H, W] ground-truth field.
        prediction: [H, W] predicted field.
        error: [H, W] error field (auto-computed if not provided).
        start: Optional start marker position (x, y, z) on sphere.
        out_dir: Directory to save figures.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    if error is None:
        error = np.abs(ground_truth - prediction)
    else:
        error = np.asarray(error)

    vmin_gt = float(np.nanmin(ground_truth))
    vmax_gt = float(np.nanmax(ground_truth))
    vmin_pred = float(np.nanmin(prediction))
    vmax_pred = float(np.nanmax(prediction))
    vmin_err = float(np.nanmin(error))
    vmax_err = float(np.nanmax(error))

    render = Sphere3DView(elev=20, azim=45)

    render.vmin, render.vmax = vmin_gt, vmax_gt
    render.render(ground_truth, start=start, title="Sphere: Ground-truth Time-to-go", out_path=f"{out_dir}/sphere_gt.png")

    render.vmin, render.vmax = vmin_pred, vmax_pred
    render.cmap = "plasma"
    render.render(prediction, start=start, title="Sphere: Predicted Time-to-go", out_path=f"{out_dir}/sphere_pred.png")

    render.vmin, render.vmax = vmin_err, vmax_err
    render.cmap = "hot"
    render.render(error, start=start, title="Sphere: Absolute Error", out_path=f"{out_dir}/sphere_error.png")


def render_torus_comparison(
    ground_truth: np.ndarray | jnp.ndarray,
    prediction: np.ndarray | jnp.ndarray,
    error: np.ndarray | jnp.ndarray | None = None,
    start: tuple[float, float] | None = None,
    obstacles: list[tuple[float, float, float]] | None = None,
    out_dir: str = "figures",
) -> None:
    """Create a three-panel comparison of ground-truth, prediction, and error on the torus.

    Args:
        ground_truth: [H, W] ground-truth field.
        prediction: [H, W] predicted field.
        error: [H, W] error field (auto-computed if not provided).
        start: Optional start marker position (θ1, θ2).
        obstacles: Optional list of (θ1, θ2, radius) obstacles.
        out_dir: Directory to save figures.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    if error is None:
        error = np.abs(ground_truth - prediction)
    else:
        error = np.asarray(error)

    vmin_gt = float(np.nanmin(ground_truth))
    vmax_gt = float(np.nanmax(ground_truth))
    vmin_pred = float(np.nanmin(prediction))
    vmax_pred = float(np.nanmax(prediction))
    vmin_err = float(np.nanmin(error))
    vmax_err = float(np.nanmax(error))

    render = Torus3DView(elev=20, azim=45)

    render.vmin, render.vmax = vmin_gt, vmax_gt
    render.render(ground_truth, start=start, obstacles=obstacles, title="Torus: Ground-truth Time-to-go", out_path=f"{out_dir}/torus_gt.png")

    render.vmin, render.vmax = vmin_pred, vmax_pred
    render.cmap = "plasma"
    render.render(prediction, start=start, obstacles=obstacles, title="Torus: Predicted Time-to-go", out_path=f"{out_dir}/torus_pred.png")

    render.vmin, render.vmax = vmin_err, vmax_err
    render.cmap = "hot"
    render.render(error, start=start, obstacles=obstacles, title="Torus: Absolute Error", out_path=f"{out_dir}/torus_error.png")


def extract_path_torus(
    field: np.ndarray,
    start: tuple[float, float],
    goal: tuple[float, float],
    num_steps: int = 50,
) -> np.ndarray:
    """Extract a path by gradient descent on the field.

    Args:
        field: [H, W] scalar field on torus (θ1, θ2 grid from -π to π).
        start: (θ1, θ2) starting position.
        goal: (θ1, θ2) goal position (for reference, not used in descent).
        num_steps: Number of gradient descent steps.

    Returns:
        [num_steps, 2] array of path points in angle space.
    """
    field = np.asarray(field)
    h, w = field.shape
    theta1_axis = np.linspace(-np.pi, np.pi, w)
    theta2_axis = np.linspace(-np.pi, np.pi, h)

    path = [np.array(start)]
    current = np.array(start, dtype=float)
    step_size = 0.02

    for _ in range(num_steps):
        # Nearest grid point
        idx1 = np.argmin(np.abs(theta1_axis - current[0]))
        idx2 = np.argmin(np.abs(theta2_axis - current[1]))

        # Compute gradient using finite differences (wrap around for torus)
        d1_plus = field[(idx2) % h, (idx1 + 1) % w] - field[idx2, idx1]
        d1_minus = field[idx2, idx1] - field[idx2, (idx1 - 1) % w]
        grad1 = (d1_plus + d1_minus) / (2 * (theta1_axis[1] - theta1_axis[0]))

        d2_plus = field[(idx2 + 1) % h, idx1] - field[idx2, idx1]
        d2_minus = field[idx2, idx1] - field[(idx2 - 1) % h, idx1]
        grad2 = (d2_plus + d2_minus) / (2 * (theta2_axis[1] - theta2_axis[0]))

        grad = np.array([grad1, grad2])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6:
            grad = grad / grad_norm

        current = current - step_size * grad

        # Wrap to [-π, π]
        current[0] = np.arctan2(np.sin(current[0]), np.cos(current[0]))
        current[1] = np.arctan2(np.sin(current[1]), np.cos(current[1]))

        path.append(current.copy())

        # Stop if we reach the goal (within tolerance)
        if np.linalg.norm(current - goal) < 0.1:
            break

    return np.array(path)


def extract_path_sphere(
    field: np.ndarray,
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    num_steps: int = 50,
) -> np.ndarray:
    """Extract a path by gradient descent on the sphere field.

    Args:
        field: [H, W] scalar field on sphere (lon/lat grid from -π to π, -π/2 to π/2).
        start: (x, y, z) starting position on unit sphere.
        goal: (x, y, z) goal position (for reference).
        num_steps: Number of gradient descent steps.

    Returns:
        [num_steps, 3] array of path points in Cartesian coordinates on sphere.
    """
    field = np.asarray(field)
    h, w = field.shape
    lon_axis = np.linspace(-np.pi, np.pi, w)
    lat_axis = np.linspace(-np.pi / 2, np.pi / 2, h)

    # Convert start point to lon/lat
    current_xyz = np.array(start)
    current_lon = np.arctan2(current_xyz[1], current_xyz[0])
    current_lat = np.arcsin(current_xyz[2])
    current = np.array([current_lon, current_lat])

    path = [current_xyz.copy()]
    step_size = 0.02

    for _ in range(num_steps):
        idx_lon = np.argmin(np.abs(lon_axis - current[0]))
        idx_lat = np.argmin(np.abs(lat_axis - current[1]))

        # Gradient
        d_lon_plus = field[idx_lat, (idx_lon + 1) % w] - field[idx_lat, idx_lon]
        d_lon_minus = field[idx_lat, idx_lon] - field[idx_lat, (idx_lon - 1) % w]
        grad_lon = (d_lon_plus + d_lon_minus) / (2 * (lon_axis[1] - lon_axis[0]))

        d_lat_plus = field[min(idx_lat + 1, h - 1), idx_lon] - field[idx_lat, idx_lon]
        d_lat_minus = field[idx_lat, idx_lon] - field[max(idx_lat - 1, 0), idx_lon]
        grad_lat = (d_lat_plus + d_lat_minus) / (2 * (lat_axis[1] - lat_axis[0]))

        grad = np.array([grad_lon, grad_lat])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6:
            grad = grad / grad_norm

        current = current - step_size * grad
        current[0] = np.arctan2(np.sin(current[0]), np.cos(current[0]))
        current[1] = np.clip(current[1], -np.pi / 2, np.pi / 2)

        # Convert back to Cartesian
        cos_lat = np.cos(current[1])
        current_xyz = np.array([cos_lat * np.cos(current[0]), cos_lat * np.sin(current[0]), np.sin(current[1])])
        path.append(current_xyz.copy())

        if np.linalg.norm(current_xyz - goal) < 0.1:
            break

    return np.array(path)


def plotly_sphere(
    field: np.ndarray | jnp.ndarray,
    start: tuple[float, float, float] | None = None,
    goal: tuple[float, float, float] | None = None,
    obstacles: list[tuple[float, float, float, float]] | None = None,
    path: np.ndarray | None = None,
    title: str = "Sphere Field",
    out_html: str | None = None,
) -> go.Figure | None:
    """Interactive Plotly 3D rendering of field on sphere (rotate/zoom with mouse).

    Args:
        field: [H, W] array of scalar values.
        start: (x, y, z) start position.
        goal: (x, y, z) goal position.
        obstacles: List of (lon, lat, radius, height_offset).
        path: [N, 3] path points in Cartesian.
        title: Plot title.
        out_html: Save interactive HTML to file if provided.

    Returns:
        Plotly Figure object (can be shown with fig.show()).
    """
    if not HAS_PLOTLY:
        print("Plotly not installed. Install with: pip install plotly")
        return None

    field = np.asarray(field)
    height, width = field.shape
    lon = np.linspace(-np.pi, np.pi, width)
    lat = np.linspace(-np.pi / 2, np.pi / 2, height)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    cos_lat = np.cos(lat_grid)
    x = cos_lat * np.cos(lon_grid)
    y = cos_lat * np.sin(lon_grid)
    z = np.sin(lat_grid)

    field_clean = np.where(np.isnan(field), np.nanmax(field), field)

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=field_clean,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Field Value"),
            )
        ]
    )

    # Add path
    if path is not None:
        path = np.asarray(path)
        fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines", name="Path", line=dict(color="cyan", width=3)))

    # Add start/goal
    if start is not None:
        fig.add_trace(go.Scatter3d(x=[start[0]], y=[start[1]], z=[start[2]], mode="markers", name="Start", marker=dict(color="red", size=10)))

    if goal is not None:
        fig.add_trace(go.Scatter3d(x=[goal[0]], y=[goal[1]], z=[goal[2]], mode="markers", name="Goal", marker=dict(color="lime", size=10)))

    # Add obstacles
    if obstacles:
        for lon_obs, lat_obs, rad_obs, height_off in obstacles:
            cos_lat_obs = np.cos(lat_obs)
            x_obs = (1.0 + height_off) * cos_lat_obs * np.cos(lon_obs)
            y_obs = (1.0 + height_off) * cos_lat_obs * np.sin(lon_obs)
            z_obs = (1.0 + height_off) * np.sin(lat_obs)
            fig.add_trace(
                go.Scatter3d(x=[x_obs], y=[y_obs], z=[z_obs], mode="markers", name="Obstacle", marker=dict(color="orange", size=8))
            )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        width=1000,
        height=900,
    )

    if out_html:
        fig.write_html(out_html)
        print(f"Saved interactive sphere visualization to {out_html}")

    return fig


def plotly_torus(
    field: np.ndarray | jnp.ndarray,
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    start: tuple[float, float] | None = None,
    goal: tuple[float, float] | None = None,
    obstacles: list[tuple[float, float, float]] | None = None,
    path: np.ndarray | None = None,
    title: str = "Torus Field",
    out_html: str | None = None,
) -> go.Figure | None:
    """Interactive Plotly 3D rendering of field on torus (rotate/zoom with mouse).

    Args:
        field: [H, W] array of scalar values.
        major_radius: Major radius of torus.
        minor_radius: Minor radius of torus.
        start: (θ1, θ2) start position.
        goal: (θ1, θ2) goal position.
        obstacles: List of (θ1, θ2, radius).
        path: [N, 2] path points in angle space.
        title: Plot title.
        out_html: Save interactive HTML to file if provided.

    Returns:
        Plotly Figure object.
    """
    if not HAS_PLOTLY:
        print("Plotly not installed. Install with: pip install plotly")
        return None

    field = np.asarray(field)
    height, width = field.shape
    theta1 = np.linspace(-np.pi, np.pi, width)
    theta2 = np.linspace(-np.pi, np.pi, height)
    theta1_grid, theta2_grid = np.meshgrid(theta1, theta2)

    R, r = major_radius, minor_radius
    x = (R + r * np.cos(theta2_grid)) * np.cos(theta1_grid)
    y = (R + r * np.cos(theta2_grid)) * np.sin(theta1_grid)
    z = r * np.sin(theta2_grid)

    field_clean = np.where(np.isnan(field), np.nanmax(field), field)

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=field_clean,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Field Value"),
            )
        ]
    )

    # Add path
    if path is not None:
        path = np.asarray(path)
        path_3d = np.zeros((len(path), 3))
        for i, (t1, t2) in enumerate(path):
            path_3d[i, 0] = (R + r * np.cos(t2)) * np.cos(t1)
            path_3d[i, 1] = (R + r * np.cos(t2)) * np.sin(t1)
            path_3d[i, 2] = r * np.sin(t2)
        fig.add_trace(go.Scatter3d(x=path_3d[:, 0], y=path_3d[:, 1], z=path_3d[:, 2], mode="lines", name="Path", line=dict(color="cyan", width=3)))

    # Add start/goal
    if start is not None:
        t1_start, t2_start = start
        x_start = (R + r * np.cos(t2_start)) * np.cos(t1_start)
        y_start = (R + r * np.cos(t2_start)) * np.sin(t1_start)
        z_start = r * np.sin(t2_start)
        fig.add_trace(
            go.Scatter3d(x=[x_start], y=[y_start], z=[z_start], mode="markers", name="Start", marker=dict(color="red", size=10))
        )

    if goal is not None:
        t1_goal, t2_goal = goal
        x_goal = (R + r * np.cos(t2_goal)) * np.cos(t1_goal)
        y_goal = (R + r * np.cos(t2_goal)) * np.sin(t1_goal)
        z_goal = r * np.sin(t2_goal)
        fig.add_trace(
            go.Scatter3d(x=[x_goal], y=[y_goal], z=[z_goal], mode="markers", name="Goal", marker=dict(color="lime", size=10))
        )

    # Add obstacles
    if obstacles:
        for t1_obs, t2_obs, rad_obs in obstacles:
            x_obs = (R + r * np.cos(t2_obs)) * np.cos(t1_obs)
            y_obs = (R + r * np.cos(t2_obs)) * np.sin(t1_obs)
            z_obs = r * np.sin(t2_obs)
            fig.add_trace(
                go.Scatter3d(x=[x_obs], y=[y_obs], z=[z_obs], mode="markers", name="Obstacle", marker=dict(color="orange", size=8))
            )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        width=1000,
        height=900,
    )

    if out_html:
        fig.write_html(out_html)
        print(f"Saved interactive torus visualization to {out_html}")

    return fig


# ---- generic ambient-grid renderer -------------------------------------------------------
#
# The classes above reconstruct the embedding from a fixed lon/lat or (theta1, theta2)
# parameterization. The functions below instead take *already-embedded* (x, y, z) grids —
# what an Environment's own `grid()` gives directly for the sphere (points are ambient unit
# vectors already) and what a torus angle grid becomes after `torus_embedding`. This is the
# path `srms/viz_3d.py` uses so the 3D figure matches the exact geometry (and obstacle/start
# placement) the environment trained against, instead of re-deriving it.


def torus_embedding(
    theta1: np.ndarray, theta2: np.ndarray, major_radius: float = 1.0, minor_radius: float = 0.35
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed flat-torus angles ``(theta1, theta2)`` into R^3 as a ring torus."""
    x = (major_radius + minor_radius * np.cos(theta2)) * np.cos(theta1)
    y = (major_radius + minor_radius * np.cos(theta2)) * np.sin(theta1)
    z = minor_radius * np.sin(theta2)
    return x, y, z


def nearest_grid_index(x: np.ndarray, y: np.ndarray, z: np.ndarray, point: tuple[float, float, float]) -> tuple[int, int]:
    """Grid cell whose ambient position is closest to ``point`` (brute-force nearest neighbor)."""
    d2 = (x - point[0]) ** 2 + (y - point[1]) ** 2 + (z - point[2]) ** 2
    i, j = np.unravel_index(np.argmin(d2), d2.shape)
    return int(i), int(j)


def extract_path_grid(
    field: np.ndarray,
    start_idx: tuple[int, int],
    goal_idx: tuple[int, int] | None = None,
    num_steps: int = 100,
    periodic: tuple[bool, bool] = (True, True),
    step_size: float = 0.6,
    goal_tol: float = 1.5,
) -> np.ndarray:
    """Gradient-descent path on ``field`` in grid-index space (float ``[i, j]`` per step).

    Index-space only — agnostic to what the axes mean geometrically, so it works identically for
    the torus's two periodic angles and the sphere's (non-periodic colatitude, periodic azimuth)
    grid via the ``periodic`` flag per axis. Map the result to 3D with ``grid_path_to_xyz``.
    """
    h, w = field.shape
    field_filled = np.where(np.isnan(field), np.nanmax(field), field)
    pos = np.array(start_idx, dtype=float)
    path = [pos.copy()]
    for _ in range(num_steps):
        i0, j0 = int(round(pos[0])) % h, int(round(pos[1])) % w
        ip = (i0 + 1) % h if periodic[0] else min(i0 + 1, h - 1)
        im = (i0 - 1) % h if periodic[0] else max(i0 - 1, 0)
        jp = (j0 + 1) % w if periodic[1] else min(j0 + 1, w - 1)
        jm = (j0 - 1) % w if periodic[1] else max(j0 - 1, 0)
        grad = np.array(
            [
                (field_filled[ip, j0] - field_filled[im, j0]) / 2.0,
                (field_filled[i0, jp] - field_filled[i0, jm]) / 2.0,
            ]
        )
        norm = np.linalg.norm(grad)
        if not np.isfinite(norm) or norm < 1e-9:
            break  # NaN gradient (e.g. a diverged/all-NaN field) is "stuck", not a step to take
        pos = pos - step_size * grad / norm
        pos[0] = pos[0] % h if periodic[0] else np.clip(pos[0], 0, h - 1)
        pos[1] = pos[1] % w if periodic[1] else np.clip(pos[1], 0, w - 1)
        path.append(pos.copy())
        if goal_idx is not None and np.linalg.norm(pos - np.array(goal_idx, dtype=float)) < goal_tol:
            break
    return np.array(path)


def grid_path_to_xyz(path_idx: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Map a grid-index path (from ``extract_path_grid``) to ambient ``[T, 3]`` positions."""
    h, w = x.shape
    pts = []
    for i, j in path_idx:
        ii, jj = int(round(i)) % h, int(round(j)) % w
        pts.append([float(x[ii, jj]), float(y[ii, jj]), float(z[ii, jj])])
    return np.array(pts)


def close_periodic(arr: np.ndarray, periodic: tuple[bool, bool]) -> np.ndarray:
    """Append a wrap-around row/column along each periodic grid axis.

    ``Environment.grid()`` covers a periodic angle as ``[start, start + 2*pi)`` — the last sample
    is one step short of closing the loop (``endpoint=False``) — so a surface plotted straight from
    it leaves a visible slit where the parameterization wraps back to its own start. Duplicating the
    first row/column onto the end closes it. No-op on axes that aren't periodic (a sphere's
    colatitude, e.g.).
    """
    if periodic[0]:
        arr = np.concatenate([arr, arr[:1]], axis=0)
    if periodic[1]:
        arr = np.concatenate([arr, arr[:, :1]], axis=1)
    return arr


def _contour_face_mask(field: np.ndarray, levels: list[float]) -> np.ndarray:
    """Boolean mask, same shape as ``field``, of grid cells adjacent to a sign change of
    ``field - level`` (in either grid direction) for any ``level`` — i.e. the cells a level's
    iso-contour actually crosses, at grid resolution. One-cell-wide regardless of the field's
    scale or gradient magnitude, unlike a fixed value-tolerance band."""
    mask = np.zeros(field.shape, dtype=bool)
    for level in levels:
        sign = np.sign(field - level)
        crosses = np.zeros_like(mask)
        row_change = sign[:-1, :] != sign[1:, :]
        crosses[:-1, :] |= row_change
        crosses[1:, :] |= row_change
        col_change = sign[:, :-1] != sign[:, 1:]
        crosses[:, :-1] |= col_change
        crosses[:, 1:] |= col_change
        mask |= crosses
    return mask


def render_ambient_mpl(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    field: np.ndarray,
    start_xyz: tuple[float, float, float] | None = None,
    obstacle_xyz: list[tuple[float, float, float]] | None = None,
    contour_levels: list[float] | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    elev: float = 20.0,
    azim: float = 45.0,
    title: str = "",
    zoom: float = 1.6,
) -> plt.cm.ScalarMappable:
    """Matplotlib counterpart of the Plotly ``render_surface``/``plotly_surface`` ambient-grid path,
    for environments (or setups) without ``plotly``/``kaleido`` installed.

    Draws onto a caller-supplied 3D ``ax`` (unlike ``Sphere3DView``/``Torus3DView``, which each own
    their figure) so several manifolds' fields can be composed as panels of one shared figure — e.g.
    ``paper_figures/intro_figure.py``'s torus/sphere/hyperbolic layout. Takes an already-embedded
    ambient grid rather than reconstructing one from an assumed parameterization, so it works
    identically for a sphere or Lorentz-hyperboloid ``env.grid()`` (already ambient) and for a torus
    grid pre-mapped through ``torus_embedding``. Returns the ``ScalarMappable`` so the caller can
    attach a colorbar (``fig.colorbar(sm, ax=ax, ...)``) if desired.

    ``contour_levels``, if given, draws white iso-value contours of the field *baked into the
    surface's own facecolors* (``_contour_face_mask``) rather than as a separate line artist.
    mplot3d has no real per-pixel depth buffer — artists are depth-sorted by centroid, not
    composited per-pixel — so a ``Line3D`` contour overlay on a self-occluding surface (a torus,
    most obviously) renders behind the surface unpredictably regardless of its ``zorder``. Baking
    the contour into the same ``Poly3DCollection`` as the field sidesteps that: each quad the
    contour crosses is recolored white, so it's depth-sorted as part of the surface itself and is
    always visible exactly where the surface itself is.
    """
    field_clean = np.where(np.isnan(field), np.nanmax(field), field)
    vmin = vmin if vmin is not None else float(np.nanmin(field_clean))
    vmax = vmax if vmax is not None else float(np.nanmax(field_clean))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = plt.get_cmap(cmap)(norm(field_clean))

    if contour_levels:
        colors = colors.copy()
        colors[_contour_face_mask(field_clean, contour_levels)] = (1.0, 1.0, 1.0, 1.0)

    ax.plot_surface(x, y, z, facecolors=colors, shade=False, rstride=1, cstride=1)

    if obstacle_xyz:
        ox, oy, oz = zip(*obstacle_xyz)
        ax.scatter(ox, oy, oz, color="orange", s=60, marker="o", edgecolors="red", linewidths=1.2, zorder=4, label="Obstacles")
    if start_xyz is not None:
        ax.scatter(*start_xyz, color="red", s=150, marker="*", edgecolors="white", linewidths=1, zorder=5, label="Start")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.view_init(elev=elev, azim=azim)
    # Proportional to the data's actual extent per axis, not a flat [1, 1, 1] — mplot3d otherwise
    # independently stretches each axis to fill the same display cube regardless of its data range,
    # which on a flat shape like the torus (a small z-extent next to a much larger x/y extent)
    # visibly exaggerates it into a tall barrel. A sphere's equal extents make this a no-op there.
    extent = (float(np.ptp(x)), float(np.ptp(y)), float(np.ptp(z)))
    # zoom > 1 shrinks mplot3d's default margin around the data within its (fixed-size) viewport —
    # with the axis off (no panes/ticks to reserve room for), most of that default margin is just
    # whitespace, so zoom in to have the surface itself fill the panel.
    ax.set_box_aspect(tuple(e if e > 1e-9 else 1.0 for e in extent), zoom=zoom)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()  # drop the 3D panes/grid/edge box entirely — just the surface itself, so
    # bbox_inches="tight" (see paper_figures/style.py's savefig) crops to the object's own
    # silhouette instead of the full axis cube.

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


def _bilinear_sample(grid: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Bilinearly sample a 2D ``grid`` at fractional ``(row, col)`` coordinates (clamped to bounds)."""
    h, w = grid.shape
    row = np.clip(row, 0, h - 1)
    col = np.clip(col, 0, w - 1)
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)
    fr, fc = row - r0, col - c0
    top = grid[r0, c0] * (1 - fc) + grid[r0, c1] * fc
    bot = grid[r1, c0] * (1 - fc) + grid[r1, c1] * fc
    return top * (1 - fr) + bot * fr


def compute_field_contours(
    field: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, levels: list[float]
) -> list[np.ndarray]:
    """Extract iso-value contour polylines of ``field`` and map them onto the ambient ``(x, y, z)``
    surface — the 3D counterpart of the white contour overlay ``srms.viz.render`` draws on the flat
    chart. Uses ``contourpy`` (matplotlib's own contour engine) on the raveled grid in index space,
    then bilinearly samples the ambient grids at each polyline's fractional ``(row, col)`` to place it
    on the curved surface. NaN cells (obstacle interiors) are skipped automatically by contourpy.
    """
    import contourpy

    cg = contourpy.contour_generator(z=field)
    polylines = []
    for level in levels:
        for line in cg.lines(level):
            col, row = line[:, 0], line[:, 1]
            xi = _bilinear_sample(x, row, col)
            yi = _bilinear_sample(y, row, col)
            zi = _bilinear_sample(z, row, col)
            polylines.append(np.stack([xi, yi, zi], axis=-1))
    return polylines


# mpl colormap name -> Plotly colorscale name, plus whether the mapped scale runs the opposite
# direction and needs `reversescale=True` to match matplotlib's own low->high color order.
_MPL_TO_PLOTLY_CMAP: dict[str, tuple[str, bool]] = {
    "viridis": ("Viridis", False),
    "plasma": ("Plasma", False),
    "inferno": ("Inferno", False),
    "hot": ("Hot", False),
    "coolwarm": ("RdBu", True),
    "bwr": ("RdBu", True),
    "rdylbu": ("RdYlBu", True),
}


def _plotly_colorscale(cmap: str) -> tuple[str, bool]:
    return _MPL_TO_PLOTLY_CMAP.get(cmap.lower(), (cmap.capitalize(), False))


def _build_surface_figure(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    field: np.ndarray,
    start_xyz: tuple[float, float, float] | None,
    goal_xyz: tuple[float, float, float] | None,
    obstacle_xyz: list[tuple[float, float, float]] | None,
    obstacle_rings: list[np.ndarray] | None,
    obstacle_patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None,
    field_contours: list[np.ndarray] | None,
    path_xyz: np.ndarray | None,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> go.Figure:
    """Shared trace-building for both the interactive (``plotly_surface``) and static
    (``render_surface``, via kaleido) exports, so the two never drift apart in what they draw."""
    field_clean = np.where(np.isnan(field), np.nanmax(field), field)
    vmin = vmin if vmin is not None else float(np.nanmin(field_clean))
    vmax = vmax if vmax is not None else float(np.nanmax(field_clean))
    colorscale, reverse = _plotly_colorscale(cmap)

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=field_clean,
                colorscale=colorscale,
                reversescale=reverse,
                cmin=vmin,
                cmax=vmax,
                showscale=True,
                colorbar=dict(title="Field Value"),
                lighting=dict(ambient=0.65, diffuse=0.5, specular=0.15, roughness=0.9),
            )
        ]
    )

    if field_contours:
        # Thick white iso-value contour lines of the field itself, mapped onto the ambient surface —
        # the 3D counterpart of the white contour overlay `srms.viz.render` draws on the flat chart.
        for i, contour in enumerate(field_contours):
            fig.add_trace(
                go.Scatter3d(
                    x=contour[:, 0], y=contour[:, 1], z=contour[:, 2], mode="lines", name="Contours",
                    legendgroup="contours", showlegend=(i == 0), line=dict(color="white", width=4),
                    hoverinfo="skip",
                )
            )

    if obstacle_patches:
        # Translucent fill of the obstacle's interior (SDF < 0), offset a hair outward along the
        # local normal so it doesn't z-fight with the coincident main field surface beneath it.
        for px, py, pz in obstacle_patches:
            fig.add_trace(
                go.Surface(
                    x=px, y=py, z=pz,
                    surfacecolor=np.zeros_like(px),
                    colorscale=[[0, "grey"], [1, "grey"]],
                    showscale=False,
                    opacity=0.45,
                    name="Obstacles",
                    legendgroup="obstacles",
                    showlegend=False,  # the boundary ring trace below carries the one legend entry
                    hoverinfo="skip",
                    lighting=dict(ambient=0.95, diffuse=0.1, specular=0.0),
                )
            )

    if path_xyz is not None and len(path_xyz) > 1:
        fig.add_trace(
            go.Scatter3d(x=path_xyz[:, 0], y=path_xyz[:, 1], z=path_xyz[:, 2], mode="lines", name="Path", line=dict(color="cyan", width=6))
        )
    if start_xyz is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[start_xyz[0]], y=[start_xyz[1]], z=[start_xyz[2]], mode="markers", name="Start",
                marker=dict(color="red", size=6, line=dict(color="white", width=1)),
            )
        )
    if goal_xyz is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[goal_xyz[0]], y=[goal_xyz[1]], z=[goal_xyz[2]], mode="markers", name="Farthest point",
                marker=dict(color="lime", size=6, line=dict(color="white", width=1)),
            )
        )
    if obstacle_rings:
        # Boundary loop (where the obstacle's SDF = 0) traced onto the ambient surface — shows the
        # obstacle's actual footprint, not just a point at its centre. One legend entry for the group.
        for i, ring in enumerate(obstacle_rings):
            fig.add_trace(
                go.Scatter3d(
                    x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="lines", name="Obstacles",
                    legendgroup="obstacles", showlegend=(i == 0), line=dict(color="dimgrey", width=5),
                )
            )
    if obstacle_xyz:
        ox, oy, oz = zip(*obstacle_xyz)
        fig.add_trace(
            go.Scatter3d(
                x=list(ox), y=list(oy), z=list(oz), mode="markers", name="Obstacles",
                legendgroup="obstacles", showlegend=not obstacle_rings,
                marker=dict(color="grey", size=4, line=dict(color="dimgrey", width=1)),
            )
        )
    return fig


def _camera_eye(elev: float, azim: float, distance: float = 1.9) -> dict:
    """Convert matplotlib's ``(elev, azim)`` (degrees) to a Plotly ``scene.camera.eye`` point,
    so the two renderers can be aimed with the same angle convention."""
    elev_r, azim_r = np.radians(elev), np.radians(azim)
    return dict(
        x=distance * np.cos(elev_r) * np.cos(azim_r),
        y=distance * np.cos(elev_r) * np.sin(azim_r),
        z=distance * np.sin(elev_r),
    )


def render_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    field: np.ndarray,
    start_xyz: tuple[float, float, float] | None = None,
    goal_xyz: tuple[float, float, float] | None = None,
    obstacle_xyz: list[tuple[float, float, float]] | None = None,
    obstacle_rings: list[np.ndarray] | None = None,
    obstacle_patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    field_contours: list[np.ndarray] | None = None,
    path_xyz: np.ndarray | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    elev: float = 20.0,
    azim: float = 45.0,
    title: str = "",
    figsize: tuple[int, int] = (10, 8),
    out_path: str | None = None,
) -> None:
    """Render a scalar field over an already-embedded ambient surface ``(x, y, z)`` to a static image.

    Renders through Plotly + kaleido rather than matplotlib's ``plot_surface``: mplot3d sorts whole
    quads by centroid depth (no real z-buffer), which on a self-occluding shape like a torus produces
    visible tearing — the far wall of the tube bleeding through as a flat patch. Plotly's WebGL
    renderer (via kaleido's headless Chromium) does correct per-pixel depth compositing instead.

    The embedding is supplied rather than reconstructed from an assumed lon/lat or angle
    parameterization — the caller (e.g. ``srms/viz_3d.py``) hands in the environment's own grid, so
    obstacles/start/path line up exactly with what training saw. ``obstacle_rings`` (each an ``[N, 3]``
    boundary loop where the obstacle's SDF = 0) draws its actual footprint; ``obstacle_xyz`` alone
    would only mark its centre. ``obstacle_patches`` (each an ``(x, y, z)`` grid over the filled
    interior, offset a hair outward to avoid z-fighting) shades that footprint translucently.
    ``field_contours`` (from ``compute_field_contours``; each an ``[N, 3]`` polyline) draws thick
    white iso-value contour lines of the field itself.
    """
    if not HAS_PLOTLY:
        print("Plotly/kaleido not installed — static 3D export skipped. Install with: pip install plotly kaleido")
        return

    fig = _build_surface_figure(
        x, y, z, field, start_xyz, goal_xyz, obstacle_xyz, obstacle_rings, obstacle_patches, field_contours,
        path_xyz, cmap, vmin, vmax,
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=_camera_eye(elev, azim)),
        ),
        showlegend=start_xyz is not None or goal_xyz is not None or path_xyz is not None,
        legend=dict(x=0.01, y=0.98, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.75)"),
        width=int(figsize[0] * 100),
        height=int(figsize[1] * 100),
    )
    if out_path:
        fig.write_image(out_path, scale=2)


def render_surface_multiangle(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    field: np.ndarray,
    out_dir: str,
    title: str = "Field",
    front: tuple[float, float] = (30.0, 0.0),
    **kwargs,
) -> None:
    """``render_surface`` from 4 camera angles (front/back/top/isometric); see that function for kwargs.

    ``front`` is ``(elev, azim)`` in degrees, default aims from ``+x`` tilted up 30°. A surface that
    wraps back on itself (a torus, most obviously) can hide an arbitrary point from any *fixed* small
    set of angles — pass ``front`` aimed at whatever the figure's "you are here" marker actually is
    (see ``srms/viz_3d.py``, which aims it at the source) so that marker is guaranteed visible in at
    least the front/back pair; ``back`` mirrors it 180° around at the same elevation.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    front_elev, front_azim = front
    angles = [
        (front_elev, front_azim, "front"),
        (front_elev, front_azim + 180.0, "back"),
        (90.0, 0.0, "top"),
        (20.0, 45.0, "isometric"),
    ]
    slug = title.lower().replace(" ", "_")
    for elev, azim, name in angles:
        render_surface(
            x, y, z, field, elev=elev, azim=azim, title=f"{title} ({name})", out_path=f"{out_dir}/{slug}_{name}.png", **kwargs
        )


def plotly_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    field: np.ndarray,
    start_xyz: tuple[float, float, float] | None = None,
    goal_xyz: tuple[float, float, float] | None = None,
    obstacle_xyz: list[tuple[float, float, float]] | None = None,
    obstacle_rings: list[np.ndarray] | None = None,
    obstacle_patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    field_contours: list[np.ndarray] | None = None,
    path_xyz: np.ndarray | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "Field",
    out_html: str | None = None,
) -> go.Figure | None:
    """Interactive (rotate/zoom/pan) Plotly rendering over an already-embedded ambient surface;
    see ``render_surface`` for the static-export counterpart sharing the same trace-building."""
    if not HAS_PLOTLY:
        print("Plotly not installed. Install with: pip install plotly")
        return None

    fig = _build_surface_figure(
        x, y, z, field, start_xyz, goal_xyz, obstacle_xyz, obstacle_rings, obstacle_patches, field_contours,
        path_xyz, cmap, vmin, vmax,
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        legend=dict(x=0.01, y=0.98, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.75)"),
        width=1000,
        height=900,
    )

    if out_html:
        fig.write_html(out_html)
        print(f"Saved interactive visualization to {out_html}")

    return fig


if __name__ == "__main__":
    import ground_truth

    # Example: render sphere
    prob = ground_truth.make_sphere(resolution=100, start_latlon_deg=(0.0, 0.0))
    field_img = prob.to_image(prob.ground_truth)
    Sphere3DView().render(field_img, start=(1, 0, 0), title="Sphere Ground-truth", out_path="figures/sphere_example.png")

    # Multi-angle sphere
    Sphere3DView().render_multiangle(field_img, start=(1, 0, 0), title="Sphere Ground-truth", out_dir="figures")

    # Interactive sphere (if Plotly available)
    plotly_sphere(field_img, start=(1, 0, 0), title="Sphere Interactive", out_html="figures/sphere_interactive.html")

    # Example: render torus
    theta1 = np.linspace(-np.pi, np.pi, 100)
    theta2 = np.linspace(-np.pi, np.pi, 100)
    t1_grid, t2_grid = np.meshgrid(theta1, theta2)
    field_torus = np.sqrt((np.cos(t1_grid)) ** 2 + (np.cos(t2_grid)) ** 2)
    Torus3DView().render(field_torus, start=(-1.5, -1.5), title="Torus Example", out_path="figures/torus_example.png")

    # Multi-angle torus
    Torus3DView().render_multiangle(field_torus, start=(-1.5, -1.5), title="Torus Example", out_dir="figures")

    # Interactive torus (if Plotly available)
    plotly_torus(field_torus, start=(-1.5, -1.5), title="Torus Interactive", out_html="figures/torus_interactive.html")
