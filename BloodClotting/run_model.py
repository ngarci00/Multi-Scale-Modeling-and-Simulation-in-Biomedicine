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
    update_particles_with_activation_and_adhesion,
)

# Parameters for the RBCs and PLTs
rbc_radius = 8.0
rbc_mass = 1.1
plt_radius = 1.5
plt_mass = 0.0124

n_rbcs = 80  #number of RBCs to simulate
n_plts = 45  #number of platelets to simulate
rng_seed = 42  #seed for reproducibility
k_contact = 1  #contact stiffness for particle-particle and particle-wall interactions
#Platelet activation and adhesion parameters
threshold = 45 #threshold for platelet activation based on contact force
activation_time_required = 2 #time required for a platelet to become fully activated after exceeding the activation threshold
adhesion_cutoff = 4 * plt_radius
k_adhesion = 3.0  #adhesion stiffness for activated platelets in contact with the damaged region

# Time stepping for the simulation
output_dt = 0.1  #time interval for recording particle states and plotting
n_steps = 3000 
dt_max = output_dt

#Vessel and flow parameters
L = 400
D = 100
R = D / 2
mu = 0.012
V_max = 1.0
inlet_width = 20.0

#Place the damaged region near the inlet so platelet activation is testable
#without requiring an extremely long simulation time.
damage_center_x = 150.0
damage_region = {
    "x_min": damage_center_x - 30.0,
    "x_max": damage_center_x + 30.0,
    "y": -R,
    "contact_y": -R + 2 * plt_radius,
}

# Initialize the random number generator
rng = np.random.default_rng(rng_seed)

# Random initial particle positions inside the analytic vessel walls
upper_bound_RBC = R - rbc_radius
lower_bound_RBC = -upper_bound_RBC
upper_bound_PLT = R - plt_radius
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

print(f"Using output_dt = {output_dt:.3f}")

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
            R,
            V_max,
            k_contact,
            damage_region,
            k_adhesion,
            dt_max,
        )
        dt = min(dt, frame_end - t)
        last_dt = dt

        update_particles_with_activation_and_adhesion(
            particles,
            dt,
            mu,
            R,
            V_max,
            k_contact,
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
            f"last solver dt = {last_dt:.3e}"
        )

# Plot particle trajectories
plt.figure(figsize=(8, 4))
plotted_labels = set()
for particle, history in zip(particles, position_history):
    history = np.array(history)
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

wall_x = np.linspace(0, L, 1000)
plt.plot(wall_x, -R * np.ones_like(wall_x), color="royalblue", linewidth=4, label="Lower Wall")
plt.plot(wall_x, R * np.ones_like(wall_x), color="royalblue", linewidth=4, label="Upper Wall")
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

# Add the animation using matplotlib's FuncAnimation
fig, ax = plt.subplots(figsize=(8, 4))
rbc_scatter = ax.scatter([], [], label="RBCs", color="red", s=50, marker="o")
inactive_plt_scatter = ax.scatter([], [], label="Inactive PLTs", color="gold", s=30, marker="o")
activated_plt_scatter = ax.scatter([], [], label="Activated PLTs", color="blue", s=30, marker="o")
ax.plot(wall_x, -R * np.ones_like(wall_x), color="firebrick", linewidth=12, label="Lower Wall")
ax.plot(wall_x, R * np.ones_like(wall_x), color="firebrick", linewidth=12, label="Upper Wall")
ax.plot(
    [damage_region["x_min"], damage_region["x_max"]],
    [damage_region["y"], damage_region["y"]],
    color="orange",
    linewidth=12,
    label="Damage Region",
)
ax.set_xlim(0, L)
ax.set_ylim(-R, R)
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
            pos = history[frame]
            rbc_positions.append(pos)
        elif activation_history[index][frame]:
            pos = history[frame]
            activated_plt_positions.append(pos)
        else:
            pos = history[frame]
            inactive_plt_positions.append(pos)

    rbc_offsets = np.array(rbc_positions) if rbc_positions else np.empty((0, 2))
    inactive_plt_offsets = np.array(inactive_plt_positions) if inactive_plt_positions else np.empty((0, 2))
    activated_plt_offsets = np.array(activated_plt_positions) if activated_plt_positions else np.empty((0, 2))

    rbc_scatter.set_offsets(rbc_offsets)
    inactive_plt_scatter.set_offsets(inactive_plt_offsets)
    activated_plt_scatter.set_offsets(activated_plt_offsets)
    ax.set_title(f"Blood Cell Animation - Step {frame}")

    return rbc_scatter, inactive_plt_scatter, activated_plt_scatter

animation = FuncAnimation(fig, update, frames=range(0, n_steps, 5), interval=50, blit=False)
save_path = os.path.join("figs", "Blood_Cell_Animation.gif")
animation.save(save_path, writer="pillow", fps=20, dpi=300)
print(f"Animation saved to {save_path}")
plt.close()
