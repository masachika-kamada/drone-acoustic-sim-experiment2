from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from IPython.display import Audio, display


def plot_room(room: pra.ShoeBox) -> None:
    room_dim = room.get_bbox()[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    room.plot(ax=ax)

    # プロット範囲を部屋の大きさに合わせる
    ax.set_xlim([0, room_dim[0]])
    ax.set_ylim([0, room_dim[1]])
    ax.set_zlim([0, room_dim[2]])
    ax.set_box_aspect(room_dim)
    plt.show()


def plot_room_views(room: pra.ShoeBox,
                    zoom_center: Optional[List[float]] = None,
                    zoom_size: Optional[float] = None) -> None:
    # Get the room dimensions from the bounding box
    room_dim = room.get_bbox()[:, 1]

    fig = plt.figure(figsize=(15, 6))
    views = [(90, -90, "Top View"), (0, -90, "Front View"), (0, 0, "Side View")]

    for i, (elev, azim, title) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        room.plot(fig=fig, ax=ax)
        ax.view_init(elev, azim)
        ax.set_title(title)
        ax.set_xlim([0, room_dim[0]])
        ax.set_ylim([0, room_dim[1]])
        ax.set_zlim([0, room_dim[2]])

        if zoom_center is not None and zoom_size is not None:
            ax.set_xlim([zoom_center[0] - zoom_size / 2, zoom_center[0] + zoom_size / 2])
            ax.set_ylim([zoom_center[1] - zoom_size / 2, zoom_center[1] + zoom_size / 2])
            ax.set_zlim([zoom_center[2] - zoom_size / 2, zoom_center[2] + zoom_size / 2])
        else:
            ax.set_xlim([0, room_dim[0]])
            ax.set_ylim([0, room_dim[1]])
            ax.set_zlim([0, room_dim[2]])

        ax.set_box_aspect(room_dim)
    plt.show()


def play_audio(audio_data: np.ndarray, fs: int) -> None:
    display(Audio(audio_data, rate=fs))


def plot_music_spectrum(doa,
                        output_dir: Optional[str] = None,
                        display: bool = False) -> None:
    estimated_angles = doa.grid.azimuth
    music_spectrum = doa.grid.values
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1, projection="polar")
    plt.polar(estimated_angles, music_spectrum)
    plt.title("MUSIC Spectrum (Polar Coordinates)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.rad2deg(estimated_angles), music_spectrum)
    plt.title("MUSIC Spectrum (Cartesian Coordinates)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(f"{output_dir}/music_spectrum.png")
    if display:
        plt.show()
    plt.close()


def plot_reverberation_wall(room: pra.Room, filename: str) -> None:
    fig, ax = plt.subplots()
    walls = room.walls
    for wall in walls:
        xs, ys = wall.corners
        absorption = wall.absorption

        if np.array(absorption).mean() == 1:
            color = "black"
        else:
            color = "red"

        ax.plot(xs, ys, color=color)

    plt.savefig(filename)
    plt.close(fig)
