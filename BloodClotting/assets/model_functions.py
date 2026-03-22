#Script containing functions to create particles and calculate forces in the blood clotting model
import numpy as np

#Function to create a generic particle dictionary with the specified parameters
def make_particle(kind, radius, mass, position, velocity=None, activated=False):
    """Create a particle dictionary with NumPy position and velocity vectors."""
    if velocity is None:
        velocity = [0.0, 0.0]

    return {
        "kind": kind,
        "radius": radius,
        "mass": mass,
        "pos": np.array(position, dtype=float),
        "vel": np.array(velocity, dtype=float),
        "activated": activated,
    }

#Function to create a red blood cell particle with the specified parameters
def make_rbc(radius, mass, position, velocity=None):
    """Create one red blood cell particle."""
    return make_particle("RBC", radius, mass, position, velocity)


#Function to calculate the plasma velocity profile across the vessel (Poiseuille flow)
def plasma_velocity_profile(y, vessel_radius, max_velocity):
    """Return the Poiseuille plasma velocity at cross-vessel location y."""
    return max_velocity * (1 - (y / vessel_radius) ** 2)

#Function to calculate the drag force on a particle in the plasma flow
def drag_force(particle, viscosity, vessel_radius, max_velocity):
    """Stoke's drag force on a particle in the plasma flow."""
    y = particle["pos"][1]
    u_fluid = np.array(
        [plasma_velocity_profile(y, vessel_radius, max_velocity), 0.0]
    )
    v_particle = particle["vel"]

    return 6 * np.pi * viscosity * particle["radius"] * (u_fluid - v_particle)

#Function to update the particle's velocity and position based on the drag force
def update_particle_drag_only(particle, dt, viscosity, vessel_radius, max_velocity):
    """Update the particle's velocity and position based on the drag force."""
    f_drag = drag_force(particle, viscosity, vessel_radius, max_velocity)
    acceleration = f_drag / particle["mass"]

    particle["vel"] = particle["vel"] + acceleration * dt
    particle["pos"] = particle["pos"] + particle["vel"] * dt

    return particle


#Temporary alias to avoid breaking earlier code while refactoring.
update_my_particle_drag = update_particle_drag_only
