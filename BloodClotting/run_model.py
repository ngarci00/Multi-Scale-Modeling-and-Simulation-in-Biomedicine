import os
#Nicolas Garcia Callejas BENG 535 - Project 3: Blood Clotting
#This script serves as the main script to run the model. it imports the needed functions, runs the model, and plots the results!
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(SCRIPT_DIR, ".mplconfig"))

import numpy as np
import matplotlib.pyplot as plt

from assets.model_functions import (
    make_rbc,
    plasma_velocity_profile,
    update_particle_drag_only,
)

#Parameters for the RBCs and PLTs
rbc_radius = 8.0 #microns
rbc_mass = 1.1 #nanograms
plt_radius = None #microns
plt_mass = None #nanograms

#Initial conditions for the platelet
#Need to double check mass and radius of PLT as is less than RBC.
platelet = {
    "kind": "PLT",
    "radius": plt_radius, #microns
    "mass": plt_mass, #nanograms
    "pos": np.array([50.0, 0.0]), #initial position
    "vel": np.array([0.0, 0.0]), #initial velocity
    "activated": False,
}

#Small set of RBCs for the drag-only simulation test
particles = [
    make_rbc(rbc_radius, rbc_mass, [10.0, 0.0]),
    make_rbc(rbc_radius, rbc_mass, [30.0, 10.0]),
    make_rbc(rbc_radius, rbc_mass, [50.0, -10.0]),
]
#Domain Parameters:
#Vessel & Flow Parameters
L = 400 #length of the vessel in microns
D = 100 #diameter of the vessel in microns
R = D / 2 #radius of the vessel in microns
mu = 0.012 * 1e5  #plasma viscosity in dyne/cm*s to ng/microns*s
V_max = 1.0 * 1000 #maximum plasma velocity in microns/s (converted from mm/s)

#Time stepping for the drag-only simulation
dt = 1e-6  #seconds
n_steps = 100 

#Run the simulation for every RBC
position_history = [[] for _ in particles] #list of lists to store position history for each particle

#For loop to run the simulation for the specified number of steps!
for step in range(n_steps):
    #Position History for plotting and
    for i, particle in enumerate(particles):
        update_particle_drag_only(particle, dt, mu, R, V_max)
        position_history[i].append(particle["pos"].copy()) #store a copy of the current position

    #Print the positions and velocities for the first few steps to verify the simulation is running correctly:
    if step % 10 == 0:  # Print every 10 steps
        print(f"Step {step}:")
        for index, particle in enumerate(particles):
            print(
                f"  RBC {index}: position = {particle['pos']}, "
                f"velocity = {particle['vel']}"
            )
#Plotting the trajectories of the RBCs
plt.figure(figsize=(8, 4))
for i, history in enumerate(position_history):
    history = np.array(history)
    plt.plot(history[:, 0], history[:, 1], label=f"RBC {i}", marker="o", markersize=10, color="red") 
plt.xlabel("x (microns)")
plt.ylabel("y (microns)")
plt.title("Trajectories of RBCs under Drag Force")
plt.xlim(0, L)
plt.ylim(-R, R)
plt.legend()
figs_dir = os.path.join(SCRIPT_DIR, "figs")
os.makedirs(figs_dir, exist_ok=True)
plt.savefig(os.path.join(figs_dir, "RBC_Trajectories.png"), dpi=200, bbox_inches="tight")
plt.close()
