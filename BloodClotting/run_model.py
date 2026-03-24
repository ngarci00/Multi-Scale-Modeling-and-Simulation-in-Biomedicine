import os
# Nicolas Garcia Callejas BENG 535 - Project 3: Blood Clotting
# This script serves as the main script to run the model, run the simulation, and plot the results.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from assets.model_functions import (
    drag_relaxation_time,
    make_plt_population,
    make_rbc_population,
    make_wall_rbc_particles,
    sample_non_overlapping_positions,
    update_particles_with_contact,
)

# Parameters for the RBCs and PLTs
rbc_radius = 8.0  # microns
rbc_mass = 1.1  # nanograms
plt_radius = 1.5  # microns
plt_mass = 0.0124  # nanograms

n_rbcs = 40  # number of RBCs to simulate
n_plts = 30  # number of platelets to simulate
rng_seed = 42  # seed for reproducibility
k_contact = 0.1  # repulsive contact spring stiffness

# Time stepping for the simulation
dt = 1e-6  # time step in seconds
n_steps = 1000  # number of simulation steps to run

# Vessel and flow parameters
L = 400  # length of the vessel in microns
D = 100  # diameter of the vessel in microns
R = D / 2  # radius of the vessel in microns
mu = 0.012 * 1e5  # plasma viscosity in ng / (micron * s)
V_max = 1.0 * 1000  # maximum plasma velocity in microns / s

#Random damage region position generato between the bounds of the vessel:
random_region= np.random.default_rng(rng_seed).uniform(0.45, 0.55) * L
damage_region = {
    "x_min": random_region - 0.05 * L,
    "x_max": random_region + 0.05 * L,
    "y": -(R - rbc_radius),
}

#Initialize the random number generator and create the wall particles
rng = np.random.default_rng(rng_seed)
wall_particles = make_wall_rbc_particles(L, R, rbc_radius, rbc_mass)

#Random initial RBC positions inside the vessel bounds
upper_bound_RBC = R - rbc_radius
lower_bound_RBC = -R + rbc_radius

rbc_positions = sample_non_overlapping_positions(n_rbcs,rbc_radius,(rbc_radius, L - rbc_radius),(lower_bound_RBC, upper_bound_RBC),rng,existing_particles=wall_particles,)
rbc_particles = make_rbc_population(rbc_radius, rbc_mass, rbc_positions, velocity=[10.0, 0.0])

#Random initial platelet positions inside the vessel bounds
plt_particles = []
if n_plts > 0:
    plt_positions = sample_non_overlapping_positions(n_plts,plt_radius,(plt_radius, L - plt_radius),(lower_bound_RBC, upper_bound_RBC),
        rng,
        existing_particles=wall_particles + rbc_particles,
    )
    plt_particles = make_plt_population(plt_radius, plt_mass, plt_positions)

#Combine RBC and PLT particles into a single list for the simulation
particles = rbc_particles + plt_particles

# Use a timestep smaller than the fastest drag relaxation timescale
if particles:
    min_relaxation_time = min(drag_relaxation_time(particle, mu) for particle in particles)
    dt = min(dt, 0.05 * min_relaxation_time)
    print(f"Using dt = {dt:.3e} s")

# Run the simulation and store particle position history
position_history = [[] for _ in particles]
for step in range(n_steps):
    update_particles_with_contact(particles, wall_particles, dt, mu, R, V_max, k_contact)

    for index, particle in enumerate(particles):
        position_history[index].append(particle["pos"].copy())

    if step % 100 == 0:
        print(f"Step {step}:")
        for index, particle in enumerate(particles):
            print(f"  {particle['kind']} {index}: position = {particle['pos']}, " f"velocity = {particle['vel']}")

#Plot particle trajectories
plt.figure(figsize=(8, 4))
plotted_labels = set()
#For loop to plot the trajectories of RBCs and PLTs:
for particle, history in zip(particles, position_history):
    history = np.array(history)
    if particle["kind"] == "RBC":
        color = "red"
        marker_size = 8
        label = "RBCs"
    else:
        color = "gold"
        marker_size = 5
        label = "PLTs"

    if label in plotted_labels:
        label = None
    else:
        plotted_labels.add(label)

    #Plotting the trajectory of each particle with markers at each time step:
    plt.plot(history[:, 0], history[:, 1], label=label, color=color, marker="o", markersize=marker_size)

wall_positions = np.array([particle["pos"] for particle in wall_particles])
plt.scatter(wall_positions[:, 0], wall_positions[:, 1], label="Wall RBCs", color="firebrick", marker="o", s=6)
plt.plot(
    [damage_region["x_min"], damage_region["x_max"]],
    [damage_region["y"], damage_region["y"]],
    color="orange",
    linewidth=6,
    label="Damage Region",
)
plt.xlabel("D (microns)")
plt.ylabel("L (microns)")
plt.title("Blood Cell Trajectories")
plt.xlim(0, L)
plt.ylim(-R, R)
plt.legend()
os.makedirs("figs", exist_ok=True)
plt.savefig(os.path.join("figs", "RBC_Trajectories.png"), dpi=300, bbox_inches="tight")
plt.close()

#Adding the animation using matplotlib's FuncAnimation
fig, ax = plt.subplots(figsize=(8, 4))
rbc_scatter = ax.scatter([], [], label="RBCs", color="red", s=30, marker="o")
plt_scatter = ax.scatter([], [], label="PLTs", color="gold", s=15, marker="o")
wall_scatter = ax.scatter(wall_positions[:, 0], wall_positions[:, 1], label="Wall RBCs", color="firebrick", s=6)
ax.plot(
    [damage_region["x_min"], damage_region["x_max"]],
    [damage_region["y"], damage_region["y"]],
    color="orange",
    linewidth=6,
    label="Damage Region",
)
ax.set_xlim(0, L)
ax.set_ylim(-R, R)
ax.set_xlabel("D (microns)")
ax.set_ylabel("L (microns)")
ax.set_title("Blood Cell Animation")
ax.legend()

visual_scale = 1000  #Scale factor to make movement more visible in the animation

#Function to update the positions of the particles in the animation at each frame:
def update(frame):
    #lists to hold the current positions of RBCs and PLTs for the animation:
    rbc_positions = []
    plt_positions = []

    for particle, history in zip(particles, position_history):
        start = history[0]
        pos = start + visual_scale * (history[frame] - start)
        if particle["kind"] == "RBC":
            rbc_positions.append(pos)
        else:
            plt_positions.append(pos)

    rbc_offsets = np.array(rbc_positions) if rbc_positions else np.empty((0, 2))
    plt_offsets = np.array(plt_positions) if plt_positions else np.empty((0, 2))

    rbc_scatter.set_offsets(rbc_offsets)
    plt_scatter.set_offsets(plt_offsets)
    ax.set_title(f"Blood Cell Animation - Step {frame}")

    return rbc_scatter, plt_scatter, wall_scatter


animation = FuncAnimation(fig, update, frames=range(0, n_steps, 10), interval=50, blit=False)
save_path = os.path.join("figs", "Blood_Cell_Animation.gif")
animation.save(save_path, writer="pillow", fps=20)
print(f"Animation saved to {save_path}")
plt.close()
