import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def wonder_view_matrix_R(lookat, up=np.array([0, 1, 0])):
    # lookat: [3], up: [3]
    # return: [3, 3]
    lookat = lookat / np.linalg.norm(lookat)
    up = up / np.linalg.norm(up)
    right = np.cross(up, lookat)
    right = right / np.linalg.norm(right)
    up = np.cross(lookat, right)
    up = up / np.linalg.norm(up)
    # concat in column-wise
    right = right.reshape(3, 1)
    up = up.reshape(3, 1)
    lookat = lookat.reshape(3, 1)
    rot_matrix = np.hstack([-right, -up, lookat])
    return rot_matrix

def camera_traj(start_point=[0., 0., 0.], step=0, num_steps=1001, radius=0.02, in_depth=0.01):
    theta = 2 * np.pi * step / num_steps
    x = start_point[0] - radius * np.sin(theta)
    y = start_point[1] + radius - radius * np.cos(theta)
    z = start_point[2] - in_depth * (-np.abs(theta - np.pi) + np.pi) / np.pi
    
    return x, y, z


def visualize_flow_as_arrows(flow, image, percentile_threshold=80, min_vectors=100, max_vectors=200, min_magnitude=5.0, alpha=0.6, dpi=100):
    # flow: [2, H, W], values in [-512, 512]
    # image: [3, H, W], values in [0, 1]
    flow = flow.permute(1, 2, 0).detach().cpu().numpy()
    image = image.permute(1, 2, 0).detach().cpu().numpy()

    h, w = flow.shape[:2]
    
    # Calculate flow magnitudes
    magnitude = np.sqrt(np.sum(flow**2, axis=2))
    
    # Initial threshold based on percentile
    threshold = np.percentile(magnitude, percentile_threshold)
    
    # Calculate number of vectors above threshold
    vectors_count = np.sum(magnitude > threshold)
    
    # Adjust sampling spacing based on number of vectors
    if vectors_count > max_vectors:
        spacing = int(np.sqrt(vectors_count / max_vectors))
    elif vectors_count < min_vectors:
        spacing = int(np.sqrt(h * w / min_vectors))
    else:
        spacing = 16  # default spacing

    y, x = np.mgrid[0:h:spacing, 0:w:spacing]

    flow_y = flow[::spacing, ::spacing, 1]
    flow_y = -1 * flow_y    # matplotlib y-axis is upward
    flow_x = flow[::spacing, ::spacing, 0]
    
    # Calculate magnitude of flow vectors
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    
    # Create mask for significant flow
    mask = magnitude > min_magnitude

    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Make the axes occupy the whole figure
    
    # Show the original image with transparency
    ax.imshow(image, alpha=alpha)

    ax.quiver(
        x[mask], y[mask],
        flow_x[mask], flow_y[mask],
        magnitude[mask],
        scale=500,
        cmap='jet',
        width=0.005,
        headwidth=8,
        headlength=10,
        headaxislength=8,
        minshaft=2,
    )

    ax.set_axis_off()
    plt.margins(0, 0)
    
    # Convert figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Get the RGBA buffer from the canvas
    buf = canvas.buffer_rgba()
    ret = np.asarray(buf)
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Convert RGBA to RGB and normalize to [0, 1]
    ret = ret[:, :, :3] / 255.0
    
    # Convert to PyTorch tensor with shape [3, H, W]
    ret = torch.from_numpy(ret.transpose(2, 0, 1)).float()

    return ret

    
