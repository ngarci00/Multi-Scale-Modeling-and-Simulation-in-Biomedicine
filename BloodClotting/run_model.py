import os
# Nicolas Garcia Callejas BENG 535 - Project 3: Blood Clotting
# This script serves as the main script to run the model, run the simulation, and plot the results.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from assets.model_functions import (
    compute_stable_dt,
    make_plt_population,
    make_rbc_population,
    make_wall_rbc_particles,
    update_particles_with_activation_and_adhesion,
)

MICRON = 1e-6
NANOGRAM = 1e-12
MM = 1e-3

# Parameters for the RBCs and PLTs
rbc_radius = 8.0 * MICRON
rbc_mass = 1.1 * NANOGRAM
plt_radius = 1.5 * MICRON
plt_mass = 0.0124 * NANOGRAM

n_rbcs = 20  #number of RBCs to simulate
n_plts = 10  #number of platelets to simulate
rng_seed = 42  #seed for reproducibility
k_contact = 100 * NANOGRAM  # N/m in SI after converting from ng/s^2
k_wall = 1e7 * NANOGRAM  # N/m in SI after converting from ng/s^2

#Platelet activation and adhesion parameters
threshold = 40 * MICRON
activation_time_required = 1e-6  #seconds
adhesion_cutoff = 1e2 * MICRON
k_adhesion = 2e5 * NANOGRAM  # N/m in SI after converting from ng/s^2

# Time stepping for the simulation
output_dt = 1e-6  #time between saved frames in seconds
n_steps = 10000  #number of saved frames
dt_max = output_dt

#Vessel and flow parameters
L = 400 * MICRON
D = 100 * MICRON
R = D / 2
mu = 0.0012  # Pa*s = kg / (m*s)
V_max = 1.0 * MM
inlet_width = 20.0 * MICRON

#Place the damaged region near the inlet so platelet activation is testable
#without requiring an extremely long simulation time.
damage_center_x = 50.0 * MICRON
damage_region = {
    "x_min": damage_center_x - 20.0 * MICRON,
    "x_max": damage_center_x + 20.0 * MICRON,
    "y": -(R - rbc_radius),
    "contact_y": -(R - rbc_radius) + rbc_radius + plt_radius,
}

# Initialize the random number generator and create the wall particles
rng = np.random.default_rng(rng_seed)
wall_particles = make_wall_rbc_particles(L, R, rbc_radius, rbc_mass)

# Random initial particle positions inside the wall-particle boundaries
wall_center_y = R - rbc_radius
upper_bound_RBC = wall_center_y - (rbc_radius + rbc_radius)
lower_bound_RBC = -upper_bound_RBC
upper_bound_PLT = wall_center_y - (rbc_radius + plt_radius)
lower_bound_PLT = -upper_bound_PLT

rbc_positions = [
    [
        rng.uniform(rbc_radius, inlet_width - rbc_radius),
        rng.uniform(lower_bound_RBC, upper_bound_RBC),
    ]
    for _ in range(n_rbcs)
]
rbc_particles = make_rbc_population(rbc_radius, rbc_mass, rbc_positions, velocity=[0.0, 0.0])

plt_particles = []
if n_plts > 0:
    plt_positions = [
        [
            rng.uniform(plt_radius, inlet_width - plt_radius),
            rng.uniform(lower_bound_PLT, upper_bound_PLT),
        ]
        for _ in range(n_plts)
    ]
    plt_particles = make_plt_population(plt_radius, plt_mass, plt_positions)

# Combine RBC and PLT particles into a single list for the simulation
particles = rbc_particles + plt_particles

print(f"Using output_dt = {output_dt:.3e} s")

# Run the simulation and store particle state history
position_history = [[] for _ in particles]
activation_history = [[] for _ in particles]
t = 0.0
for step in range(n_steps):
    frame_end = (step + 1) * output_dt
    last_dt = dt_max

    while t < frame_end:
        dt = compute_stable_dt(
            particles,
            mu,
            k_contact,
            k_wall,
            k_adhesion,
            dt_max,
        )
        dt = min(dt, frame_end - t)
        last_dt = dt

        update_particles_with_activation_and_adhesion(
            particles,
            wall_particles,
            dt,
            mu,
            R,
            V_max,
            k_contact,
            k_wall,
            damage_region,
            threshold,
            activation_time_required,
            adhesion_cutoff,
            k_adhesion,
        )
        t += dt

    for index, particle in enumerate(particles):
        position_history[index].append(particle["pos"].copy())
        activation_history[index].append(particle["activated"])

    if step % 100 == 0:
        activated_platelets = sum(
            1 for particle in particles if particle["kind"] == "PLT" and particle["activated"]
        )
        print(
            f"Frame {step}: activated platelets = {activated_platelets}, "
            f"last solver dt = {last_dt:.3e} s"
        )

# Plot particle trajectories
plt.figure(figsize=(8, 4))
plotted_labels = set()
for particle, history in zip(particles, position_history):
    history = np.array(history) / MICRON
    if particle["kind"] == "RBC":
        color = "red"
        marker_size = 8
        label = "RBCs"
    elif particle["activated"]:
        color = "lime"
        marker_size = 5
        label = "Activated PLTs"
    else:
        color = "gold"
        marker_size = 5
        label = "Inactive PLTs"

    if label in plotted_labels:
        label = None
    else:
        plotted_labels.add(label)

    plt.plot(history[:, 0], history[:, 1], label=label, color=color, marker="o", markersize=marker_size)

wall_positions = np.array([particle["pos"] for particle in wall_particles]) / MICRON
plt.scatter(wall_positions[:, 0], wall_positions[:, 1], label="Wall RBCs", color="firebrick", marker="o", s=6)
plt.plot(
    [damage_region["x_min"] / MICRON, damage_region["x_max"] / MICRON],
    [damage_region["y"] / MICRON, damage_region["y"] / MICRON],
    color="orange",
    linewidth=6,
    label="Damage Region",
)
plt.xlabel("D (microns)")
plt.ylabel("L (microns)")
plt.title("Blood Cell Trajectories")
plt.xlim(0, L / MICRON)
plt.ylim(-R / MICRON, R / MICRON)
plt.legend()
os.makedirs("figs", exist_ok=True)
plt.savefig(os.path.join("figs", "RBC_Trajectories.png"), dpi=300, bbox_inches="tight")
plt.close()

# Add the animation using matplotlib's FuncAnimation
fig, ax = plt.subplots(figsize=(8, 4))
rbc_scatter = ax.scatter([], [], label="RBCs", color="red", s=30, marker="o")
inactive_plt_scatter = ax.scatter([], [], label="Inactive PLTs", color="gold", s=15, marker="o")
activated_plt_scatter = ax.scatter([], [], label="Activated PLTs", color="lime", s=15, marker="o")
wall_scatter = ax.scatter(wall_positions[:, 0], wall_positions[:, 1], label="Wall RBCs", color="firebrick", s=6)
ax.plot(
    [damage_region["x_min"] / MICRON, damage_region["x_max"] / MICRON],
    [damage_region["y"] / MICRON, damage_region["y"] / MICRON],
    color="orange",
    linewidth=6,
    label="Damage Region",
)
ax.set_xlim(0, L / MICRON)
ax.set_ylim(-R / MICRON, R / MICRON)
ax.set_xlabel("D (microns)")
ax.set_ylabel("L (microns)")
ax.set_title("Blood Cell Animation")
ax.legend()

#Function to update the positions of the particles in the animation at each frame
def update(frame):
    rbc_positions = []
    inactive_plt_positions = []
    activated_plt_positions = []

    for index, (particle, history) in enumerate(zip(particles, position_history)):
        if particle["kind"] == "RBC":
            pos = history[frame] / MICRON
            rbc_positions.append(pos)
        elif activation_history[index][frame]:
            pos = history[frame] / MICRON
            activated_plt_positions.append(pos)
        else:
            pos = history[frame] / MICRON
            inactive_plt_positions.append(pos)

    rbc_offsets = np.array(rbc_positions) if rbc_positions else np.empty((0, 2))
    inactive_plt_offsets = np.array(inactive_plt_positions) if inactive_plt_positions else np.empty((0, 2))
    activated_plt_offsets = np.array(activated_plt_positions) if activated_plt_positions else np.empty((0, 2))

    rbc_scatter.set_offsets(rbc_offsets)
    inactive_plt_scatter.set_offsets(inactive_plt_offsets)
    activated_plt_scatter.set_offsets(activated_plt_offsets)
    ax.set_title(f"Blood Cell Animation - Step {frame}")

    return rbc_scatter, inactive_plt_scatter, activated_plt_scatter, wall_scatter


animation = FuncAnimation(fig, update, frames=range(0, n_steps, 10), interval=50, blit=False)
save_path = os.path.join("figs", "Blood_Cell_Animation.gif")
animation.save(save_path, writer="pillow", fps=20)
print(f"Animation saved to {save_path}")
plt.close()
