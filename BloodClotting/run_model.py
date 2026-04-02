import os
#Nicolas Garcia Callejas BENG 535 - Project 3: Blood Clotting
#This script serves as the main script to run the model, run the simulation, and plot the results.
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
rbc_radius = 8.0 #RBC radius in microns
rbc_mass = 1.1 #RBC mass in nanograms
plt_radius = 1.5 #PLT radius in microns
plt_mass = 0.0124 #PLT mass in nanograms

n_rbcs = 40  #number of RBCs to simulate
n_plts = 15  #number of platelets to simulate
rng_seed = 42  #seed for reproducibility
k_contact = 3  #contact stiffness for particle-particle and particle-wall interactions
#Platelet activation and adhesion parameters
threshold = 45 #threshold for platelet activation based on contact force
activation_time_required = 2 #time required for a platelet to become fully activated after exceeding the activation threshold
adhesion_cutoff = 4 * plt_radius
k_adhesion = 3.0  #adhesion stiffness for activated platelets in contact with the damaged region

#Time stepping for the simulation
output_dt = 0.1  #time interval for recording particle states and plotting
n_steps = 6000 #number of simulation steps to run
dt_max = output_dt

#Vessel and flow parameters
L = 400 #length of the vessel in microns
D = 100 #diameter of the vessel in microns
R = D / 2 #vessel radius in microns
mu = 0.012 #dynamic viscosity of blood in g/(micron*ms)
V_max = 1.0 #maximum flow velocity in microns/ms
inlet_width = 20.0 #width of the inlet region where particles are initialized in microns

#Place the damaged region near the inlet so platelet activation is testable
#without requiring an extremely long simulation time.
damage_center_x = 150.0
damage_region = {
    "x_min": damage_center_x - 30.0,
    "x_max": damage_center_x + 30.0,
    "y": -R,
    "contact_y": -R + 2 * plt_radius,
}

#Initialize the random number generator
rng = np.random.default_rng(rng_seed)

#Random initial particle positions inside the analytic vessel walls
upper_bound_RBC = R - rbc_radius
lower_bound_RBC = -upper_bound_RBC
upper_bound_PLT = R - plt_radius
lower_bound_PLT = -upper_bound_PLT

#Generate RBC particles with random initial positions and zero initial velocity
rbc_positions = [
    [
        rng.uniform(rbc_radius, inlet_width - rbc_radius),
        rng.uniform(lower_bound_RBC, upper_bound_RBC),
    ]
    for _ in range(n_rbcs)
]
rbc_particles = make_rbc_population(rbc_radius, rbc_mass, rbc_positions, velocity=[0.0, 0.0])
#empty list to hold PLT particles
plt_particles = []
#Generate PLT particles with random initial positions and zero initial velocity
if n_plts > 0:
    plt_positions = [
        [
            rng.uniform(plt_radius, inlet_width - plt_radius),
            rng.uniform(lower_bound_PLT, upper_bound_PLT),
        ]
        for _ in range(n_plts)
    ]
    plt_particles = make_plt_population(plt_radius, plt_mass, plt_positions)

#Combine RBC and PLT particles into a single list for the simulation
particles = rbc_particles + plt_particles

#Function to recycle outflow particles back to the inlet with random y-positions and reset their state
def recycle_outflow_particles(particles, rng):
    #for loop to check each particle and recycle those that have flowed past the outlet back to the inlet in random positions!:
    for particle in particles:
        if particle["pos"][0] <= L:
            continue

        radius = particle["radius"]
        particle["pos"] = np.array(
            [
                rng.uniform(radius, inlet_width - radius),
                rng.uniform(-(R - radius), R - radius),
            ],
            dtype=float,
        )
        particle["vel"] = np.zeros(2)
        particle["activated"] = False
        particle["activation_time"] = 0.0
        particle["adhered"] = False

#print the parameters to the console for reference
print(f"\nSimulation parameters:")
print(f"Number of RBCs: {n_rbcs}")
print(f"Number of PLTs: {n_plts}")
print(f"Using output_dt = {output_dt:.3f}\n")

#Run the simulation and store particle state history
position_history = [[] for _ in particles]
activation_history = [[] for _ in particles]
t = 0.0
#Main simulation loop with adaptive time stepping based on stability criteria and platelet activation dynamics:
for step in range(n_steps):
    frame_end = (step + 1) * output_dt
    last_dt = dt_max

    while t < frame_end:
        #calling function to compute a stable time step inside the inner loop as recommended:
        dt = compute_stable_dt(particles,R,V_max,k_contact,damage_region,k_adhesion,dt_max)
        dt = min(dt, frame_end - t)
        last_dt = dt

        #calling function to update particle positions and velocities based on forces: flow, contact, activation, and adhesion:
        update_particles_with_activation_and_adhesion(particles,dt,mu,R,V_max,k_contact,damage_region,threshold,activation_time_required,adhesion_cutoff,k_adhesion)

        t += dt #increment the simulation time by the chosen time step

    #function to recycle outflow particles back to the inlet with random y-positions and reset their state:
    recycle_outflow_particles(particles, rng)

    #for loop to record the position and activation state of each particle at the end of the current frame for plotting and animation:  
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

#Add the animation using matplotlib's FuncAnimation
fig, ax = plt.subplots(figsize=(8, 4))
rbc_scatter = ax.scatter([], [], label="RBCs", color="red", s=50, marker="o")
inactive_plt_scatter = ax.scatter([], [], label="Inactive PLTs", color="gold", s=30, marker="o")
activated_plt_scatter = ax.scatter([], [], label="Activated PLTs", color="blue", s=30, marker="o")
wall_x = np.linspace(0, L, 1000)
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

    #for loop to separate the positions of RBCs, activated PLTs, and inactive PLTs for plotting in the animation:
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

#Create the animation and save it as a GIF using FuncAnimation:
animation = FuncAnimation(fig, update, frames=range(0, n_steps, 5), interval=50, blit=False)
save_path = os.path.join("figs", "Blood_Cell_Animation.gif")
animation.save(save_path, writer="pillow", fps=20, dpi=300)
print(f"Animation saved to {save_path}")
plt.close()
