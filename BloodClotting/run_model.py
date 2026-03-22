import os
#Nicolas Garcia Callejas BENG 535 - Project 3: Blood Clotting
#This script serves as the main script to run the model. it imports the needed functions, runs the model, and plots the results!
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import matplotlib.pyplot as plt
from assets.model_functions import (drag_relaxation_time,make_plt,make_plt_population,make_wall_rbc_particles,make_rbc_population,sample_non_overlapping_positions,update_particles_with_adhesion,)

#Parameters for the RBCs and PLTs
rbc_radius = 8.0 #microns
rbc_mass = 1.1 #nanograms
plt_radius = 1.5 #microns
plt_mass = 0.0124 #nanograms

n_rbcs = 40 #number of RBCs to simulate
n_plts = 20#number of platelets to simulate
rng_seed = 42 #seed for reproducibility
k_contact = 0.1 #contact spring stiffness, high k_contact means less overlap between particles,low k_contact means more overlap allowed.
k_adhesion = 0.8 #adhesion spring stiffness for platelets <- adhest for sensitivity on PLTs adhesion strength

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
damage_region = {"x_min": 0.45 * L,"x_max": 0.55 * L,"y": -(R - rbc_radius),} #damage region on the wall where platelets can adhere
adhesion_cutoff = 8 * plt_radius #distance within which platelets can adhere to the damaged wall

#Initialize the random number generator and create the wall particles:
rng = np.random.default_rng(rng_seed)
wall_particles = make_wall_rbc_particles(L, R, rbc_radius, rbc_mass)

#Random initial positions inside the vessel bounds:
rbc_positions = sample_non_overlapping_positions(n_rbcs,rbc_radius,(rbc_radius, L - rbc_radius),(-R + rbc_radius, R - rbc_radius),rng,existing_particles=wall_particles)
#Create the RBC particles based on the sampled positions:
rbc_particles = make_rbc_population(rbc_radius, rbc_mass, rbc_positions)

plt_particles = [] #Empty list to hold PLT particles
if n_plts > 0:
    if plt_radius is None or plt_mass is None:
        raise ValueError("Set platelet radius and mass before simulating platelets.")
    #This function ensures that the initial positions of the PLTs don't overlap with each other or with the wall and RBC particles:
    plt_positions = sample_non_overlapping_positions(n_plts,plt_radius,(plt_radius, L - plt_radius),(-R + plt_radius, R - plt_radius),rng,existing_particles=wall_particles + rbc_particles)
    plt_particles = make_plt_population(plt_radius, plt_mass, plt_positions)

#Combining RBC and PLT particles into a single list for the simulation:
particles = rbc_particles + plt_particles

#Use a timestep smaller than the fastest drag relaxation timescale.
if particles:
    #Calculate the minimum drag relaxation time across all particles to ensure numerical stability:
    min_relaxation_time = min(drag_relaxation_time(particle, mu) for particle in particles)
    dt = min(dt, 0.05 * min_relaxation_time)
    print(f"Using dt = {dt:.3e} s")

#Run the simulation for every RBC
position_history = [[] for _ in particles] #list of lists to store position history for each particle

#For loop to run the simulation for the specified number of steps!
for step in range(n_steps):
    update_particles_with_adhesion(particles,wall_particles,dt,mu,R,V_max,k_contact,damage_region,k_adhesion,adhesion_cutoff)

    #Position History for plotting and analysis
    for i, particle in enumerate(particles):
        position_history[i].append(particle["pos"].copy()) #store a copy of the current position

    #Print the positions and velocities for the first few steps to verify the simulation is running correctly:
    if step % 10 == 0:  #Print every 10 steps
        activated_platelets = sum(
            1 for particle in particles if particle["kind"] == "PLT" and particle["activated"]
        )
        print(f"Step {step}:")
        for index, particle in enumerate(particles):
            print(
                f"  {particle['kind']} {index}: position = {particle['pos']}, "
                f"velocity = {particle['vel']}, "
                f"activated = {particle['activated']}"
            )
        print(f"  Activated platelets: {activated_platelets}") #Print the number of activated platelets at this step

#Plotting the trajectories of the RBCs
plt.figure(figsize=(8, 4))
plotted_labels = set()
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

    plt.plot(history[:, 0],history[:, 1],label=label,color=color,marker="o",markersize=marker_size)
wall_positions = np.array([particle["pos"] for particle in wall_particles])
plt.scatter(wall_positions[:, 0],wall_positions[:, 1],label="Wall RBCs",color="firebrick",marker="o",s=6)
plt.plot([damage_region["x_min"], damage_region["x_max"]],[damage_region["y"], damage_region["y"]],color="orange",linewidth=6,label="Damage Region")
plt.xlabel("D (microns)")
plt.ylabel("L (microns)")
plt.title("Platelet Aggregation Model: RBC Trajectories")
plt.xlim(0, L)
plt.ylim(-R, R)
plt.legend()
os.makedirs("figs", exist_ok=True)
plt.savefig(os.path.join("figs", "RBC_Trajectories.png"), dpi=200, bbox_inches="tight")
plt.close()
