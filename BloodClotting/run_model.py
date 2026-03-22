import os
#Nicolas Garcia Callejas BENG 535 - Project 3: Blood Clotting
#This script serves as the main script to run the model. it imports the needed functions, runs the model, and plots the results!
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import matplotlib.pyplot as plt

from assets.model_functions import (
    make_plt,
    make_plt_population,
    make_wall_rbc_particles,
    make_rbc_population,
    sample_non_overlapping_positions,
    update_particles_with_contact,
)

#Parameters for the RBCs and PLTs
rbc_radius = 8.0 #microns
rbc_mass = 1.1 #nanograms
plt_radius = None #microns
plt_mass = None #nanograms

n_rbcs = 10 #number of RBCs to simulate
n_plts = 0 #number of platelets to simulate
rng_seed = 42 #seed for reproducibility
k_contact = 0.1 #contact spring stiffness

#Time stepping for the drag-only simulation
dt = 1e-6  #seconds
n_steps = 100 

#Platelet template for later use once parameters are confirmed
platelet = make_plt(plt_radius, plt_mass, [50.0, 0.0])

#Domain Parameters:
#Vessel & Flow Parameters
L = 400 #length of the vessel in microns
D = 100 #diameter of the vessel in microns
R = D / 2 #radius of the vessel in microns
mu = 0.012 * 1e5  #plasma viscosity in dyne/cm*s to ng/microns*s
V_max = 1.0 * 1000 #maximum plasma velocity in microns/s (converted from mm/s)

rng = np.random.default_rng(rng_seed)
wall_particles = make_wall_rbc_particles(L, R, rbc_radius, rbc_mass)

#Random initial positions inside the vessel bounds
rbc_positions = sample_non_overlapping_positions(
    n_rbcs,
    rbc_radius,
    (rbc_radius, L - rbc_radius),
    (-R + rbc_radius, R - rbc_radius),
    rng,
    existing_particles=wall_particles,
)
rbc_particles = make_rbc_population(rbc_radius, rbc_mass, rbc_positions)

plt_particles = []
if n_plts > 0:
    if plt_radius is None or plt_mass is None:
        raise ValueError("Set platelet radius and mass before simulating platelets.")

    plt_positions = sample_non_overlapping_positions(
        n_plts,
        plt_radius,
        (plt_radius, L - plt_radius),
        (-R + plt_radius, R - plt_radius),
        rng,
        existing_particles=wall_particles + rbc_particles,
    )
    plt_particles = make_plt_population(plt_radius, plt_mass, plt_positions)

particles = rbc_particles + plt_particles

#Run the simulation for every RBC
position_history = [[] for _ in particles] #list of lists to store position history for each particle

#For loop to run the simulation for the specified number of steps!
for step in range(n_steps):
    update_particles_with_contact(particles,wall_particles,dt,mu,R,V_max,k_contact,)

    #Position History for plotting and analysis
    for i, particle in enumerate(particles):
        position_history[i].append(particle["pos"].copy()) #store a copy of the current position

    #Print the positions and velocities for the first few steps to verify the simulation is running correctly:
    if step % 10 == 0:  #Print every 10 steps
        print(f"Step {step}:")
        for index, particle in enumerate(particles):
            print(
                f"  {particle['kind']} {index}: position = {particle['pos']}, "
                f"velocity = {particle['vel']}"
            )
#Plotting the trajectories of the RBCs
plt.figure(figsize=(8, 4))
for i, history in enumerate(position_history):
    history = np.array(history)
    plt.plot(history[:, 0], history[:, 1], label="RBCs", color="red", marker='o', markersize=10)  
wall_positions = np.array([particle["pos"] for particle in wall_particles])
plt.scatter(wall_positions[:, 0], wall_positions[:, 1], label="Wall RBCs", color="red", marker='o', s=36)
plt.xlabel("D (microns)")
plt.ylabel("L (microns)")
plt.title("Platelet Aggregation Model: RBC Trajectories")
plt.xlim(0, L)
plt.ylim(-R, R)
os.makedirs("figs", exist_ok=True)
plt.savefig(os.path.join("figs", "RBC_Trajectories.png"), dpi=200, bbox_inches="tight")
plt.close()
