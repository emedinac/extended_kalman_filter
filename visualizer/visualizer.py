from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pymap3d as pm
import matplotlib

matplotlib.use('Qt5Agg')  # have issues on cluster
font = {'weight': 'bold',
        'size': 10}

matplotlib.rc('font', **font)


def plot_route(mus_, vars_, xyz, measured_xyz, std=3, filename="image.png", figsize=(12, 12)):
    stdx = std * np.sqrt(vars_[:, 0, 0])
    stdy = std * np.sqrt(vars_[:, 1, 1])
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(measured_xyz[:, 0], measured_xyz[:, 1], color='c', marker='.', s=1, label='measured trajectory')
    ax.plot(xyz[:, 0], xyz[:, 1], c='g', lw=3, label='GT trajectory')
    ax.plot(mus_[:, 0], mus_[:, 1], c='r', lw=2, label='estimated trajectory')
    ax.plot(xyz[:, 0] + stdx, xyz[:, 1] + stdy, lw=2., label=f'{std}-sigma error interval', color='orange')
    ax.plot(xyz[:, 0] - stdx, xyz[:, 1] - stdy, lw=2., color='orange')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend()
    ax.grid()
    fig.set_size_inches(figsize)
    fig.set_dpi(200)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(filename), bbox_inches=extent)


class PlotOnMap():
    def __init__(self, lla, measured_xyz, folderpath="output/"):
        x, y, z = measured_xyz.T
        lat0, lon0, h0 = lla[0]
        self.measured_lla = np.array(pm.enu2geodetic(x, y, z, lat0, lon0, h0)).T[:, :2]
        idx_min = np.linalg.norm(self.measured_lla, axis=1).argmin()
        idx_max = np.linalg.norm(self.measured_lla, axis=1).argmax()
        self.valid_area = np.array([self.measured_lla[idx_min], self.measured_lla[idx_max]])
        center_lla = self.valid_area.mean(0)
        radius_km = np.linalg.norm((measured_xyz[idx_max][:2] - measured_xyz[idx_min][:2]))
        self.valid_area = self.valid_area.flatten()
        self.graph = ox.graph_from_point((center_lla[1], center_lla[0]), dist=radius_km, network_type="all")
        self.folderpath = Path(folderpath)
        self.folderpath.mkdir(parents=True, exist_ok=True)

    def _generate_image(self, filename="map.png", figsize=(8, 8)):
        fig, ax = ox.plot_graph(self.graph, node_size=10, bgcolor='k')
        fig.set_size_inches(figsize)
        fig.set_dpi(500)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, bbox_inches=extent)
        print(f'valid area defined for visualization: {self.valid_area}')

    def update_graph(self):
        for node in range(len(self.measured_lla)):
            self.graph.add_node(node, x=self.measured_lla[node][0], y=self.measured_lla[node][1])
            if node > 0:
                self.graph.add_edge(node, node - 1)

    # This function is not working well.
    def visualize_on_map(self, filename):
        central_position = [self.valid_area[::2].mean().tolist(), self.valid_area[1::2].mean().tolist()]
        map = folium.Map()
        map.fit_bounds([self.valid_area[:2][::-1].tolist(), self.valid_area[2:][::-1].tolist()])

        for node in list(self.graph.nodes):
            self.graph.remove_node(node)

        self.update_graph()

        for node in list(self.graph.nodes):
            location = [self.graph.nodes[node]["y"], self.graph.nodes[node]["x"]]
            folium.CircleMarker(location=location).add_to(map)
        folium.CircleMarker(location=central_position[::-1], color="green", fill_color="green").add_to(map)

        # route = np.arange(len(self.measured_lla)).tolist()
        # ox.plot_route_folium(self.graph, route, map, color="#ff0000", weight=5, opacity=0.6)
        nodes, edges = ox.graph_to_gdfs(self.graph)
        for _, row in edges.iterrows():
            p = np.array(row["geometry"].coords).flatten()
            folium.PolyLine(
                locations=[[p[1], p[0]], [p[3], p[2]]],
                color="red",
                weight=2,
                opacity=0.7,
            ).add_to(map)
        map.save(str(self.folderpath.joinpath(filename)))  # Map is not corresponding the openstreetmap
        print("Map (HTML) created for map visualization")
