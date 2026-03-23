import os
#Nicolas Garcia Callejas BENG 535 - Project 3: Blood Clotting
#This script serves as the main script to run the model. it imports the needed functions, runs the model, and plots the results!
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import matplotlib.pyplot as plt
from assets.model_functions import (drag_relaxation_time,make_plt,make_plt_population,make_wall_rbc_particles,make_rbc_population,sample_non_overlapping_positions,update_particles_with_adhesion,)
from matplotlib.animation import FuncAnimation

#Parameters for the RBCs and PLTs
rbc_radius = 8.0 #microns
rbc_mass = 1.1 #nanograms

plt_radius = 1.5 #microns
plt_mass = 0.0124 #nanograms

n_rbcs = 40 #number of RBCs to simulate
n_plts = 30 #number of platelets to simulate
rng_seed = 42 #seed for reproducibility
k_contact = 0.1 #contact spring stiffness, high k_contact means less overlap between particles,low k_contact means more overlap allowed.
k_adhesion = 1.0 #adhesion spring stiffness for platelets <- adhest for sensitivity on PLTs adhesion strength

#Time stepping for the drag-only simulation
dt = 1e-6  #time step in seconds
n_steps = 1000 #number of simulation steps to run

#Platelet template for later use once parameters are confirmed
platelet = make_plt(plt_radius, plt_mass, [50.0, 0.0], activated=True, adhered=True)

#Domain Parameters:
#Vessel & Flow Parameters
L = 400 #length of the vessel in microns
D = 100 #diameter of the vessel in microns
R = D / 2 #radius of the vessel in microns
mu = 0.012 * 1e5  #plasma viscosity in dyne/cm*s to ng/microns*s
V_max = 1.0 * 1000 #maximum plasma velocity in microns/s (converted from mm/s)
damage_region = {
    "x_min": 0.45 * L,
    "x_max": 0.55 * L,
    "y": -(R - rbc_radius),
    "contact_y": -(R - rbc_radius) + rbc_radius + plt_radius,
    "capture_distance": plt_radius,
} #damage region on the wall where platelets can adhere
adhesion_cutoff = 4 * plt_radius #distance within which platelets can adhere to the damaged wall

#Initialize the random number generator and create the wall particles:
rng = np.random.default_rng(rng_seed)
wall_particles = make_wall_rbc_particles(L, R, rbc_radius, rbc_mass)

#Function to check if a candidate particle would overlap with any existing particles
#candidate particle is the new particle we want to place, other_particles is the list of existing particles we want to check against:
def overlaps_existing_particles(candidate_position, candidate_radius, other_particles):
    candidate_position = np.array(candidate_position, dtype=float)

    for other_particle in other_particles:
        min_distance = candidate_radius + other_particle["radius"]
        if np.linalg.norm(candidate_position - other_particle["pos"]) < min_distance:
            return True

    return False

#Random initial positions inside the vessel bounds:
rbc_positions = sample_non_overlapping_positions(n_rbcs,rbc_radius,(rbc_radius, L - rbc_radius),(-R + rbc_radius, R - rbc_radius),rng,existing_particles=wall_particles)
#Create the RBC particles based on the sampled positions:
rbc_particles = make_rbc_population(rbc_radius, rbc_mass, rbc_positions,fixed=False, velocity=[10.0, 0.0]) #RBCs start with zero velocity and are free to move

plt_particles = [] #Empty list to hold PLT particles

if n_plts > 0:
    #Seed one platelet near the damaged wall so the adhesion model is exercised.
    minimum_wall_clearance = rbc_radius + plt_radius + 0.5
    maximum_activation_distance = adhesion_cutoff - 0.5

    seeded_platelet_y = damage_region["contact_y"] 

    seeded_platelet_positions = [] #List to hold the position of the seeded platelet near the damaged region

    candidate_x_positions = np.linspace(
        damage_region["x_min"] + plt_radius,
        damage_region["x_max"] - plt_radius,
        41,
    )

    for x_position in candidate_x_positions:
        candidate_position = [x_position, seeded_platelet_y]
        if not overlaps_existing_particles(candidate_position, plt_radius, wall_particles + rbc_particles):
            seeded_platelet_positions.append(candidate_position)
            break

    if not seeded_platelet_positions:
        raise ValueError(
            "Could not place a platelet near the damaged region without overlap."
        )

    #This function ensures that the remaining PLTs don't overlap with each other,
    #the wall particles, the RBCs, or the seeded near-damage platelet.
    plt_positions = list(seeded_platelet_positions)
    remaining_plts = n_plts - len(seeded_platelet_positions)
    if remaining_plts > 0:
        plt_positions.extend(
            sample_non_overlapping_positions(
                remaining_plts,
                plt_radius,
                (plt_radius, L - plt_radius),
                (-R + plt_radius, R - plt_radius),
                rng,
                existing_particles=wall_particles
                + rbc_particles
                + make_plt_population(plt_radius, plt_mass, seeded_platelet_positions),
            )
        )

    plt_particles = make_plt_population(plt_radius, plt_mass, plt_positions,activated=False, adhered=False) #PLTs start activated and adhered to test the adhesion model right away

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
        adhered_platelets = sum(
            1 for particle in particles if particle["kind"] == "PLT" and particle["adhered"]
        )
        print(f"Step {step}:")
        for index, particle in enumerate(particles):
            print(
                f"  {particle['kind']} {index}: position = {particle['pos']}, "
                f"velocity = {particle['vel']}, "
                f"activated = {particle['activated']}, "
                f"adhered = {particle['adhered']}"
            )
        print(f"  Activated platelets: {activated_platelets}") #Print the number of activated platelets at this step
        print(f"  Adhered platelets: {adhered_platelets}")

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
wall_positions = np.array([particle["pos"] for particle in wall_particles]) #Plot the wall particles as a reference for the vessel walls
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

#Adding the animation using matplotlib's FuncAnimation
fig, ax = plt.subplots(figsize=(8, 4))
rbc_scatter = ax.scatter([], [], label="RBCs", color="red", s=30, marker="o")
plt_scatter = ax.scatter([], [], label="PLTs", color="gold", s=15, marker="o")
wall_scatter = ax.scatter(wall_positions[:, 0], wall_positions[:, 1], label="Wall RBCs", color="firebrick", s=6)
ax.plot([damage_region["x_min"], damage_region["x_max"]],[damage_region["y"], damage_region["y"]],color="orange",linewidth=6,label="Damage Region")
ax.set_xlim(0, L)
ax.set_ylim(-R, R)
ax.set_xlabel("D (microns)")
ax.set_ylabel("L (microns)")
ax.set_title("Platelet Aggregation Model Animation")
ax.legend()

def update(frame):
    rbc_positions = []
    plt_positions = []
    visual_scale = 1000

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
    ax.set_title(f"Platelet Aggregation Model Animation - Step {frame}")

    return rbc_scatter, plt_scatter, wall_scatter

animation = FuncAnimation(fig, update, frames=range(0, n_steps, 10), interval=50, blit=False)
save_path = os.path.join("figs", "Platelet_Aggregation_Animation.gif")
animation.save(save_path, writer="pillow", fps=20)
print(f"Animation saved to {save_path}")
plt.close()
