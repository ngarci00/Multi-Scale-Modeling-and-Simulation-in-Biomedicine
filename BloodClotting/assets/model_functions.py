#Nicolas Garcia Callejas - BENG 535 - Project 3: Blood Clotting
#This script serves as a collection of all the used functions needed in the main script to run the model:
import numpy as np

# Function to create a generic particle dictionary with the specified parameters
def make_particle(kind, radius, mass, position, velocity=None, fixed=False):
    if velocity is None:
        velocity = [0.0, 0.0]

    return {
        "kind": kind,
        "radius": radius,
        "mass": mass,
        "pos": np.array(position, dtype=float),
        "vel": np.array(velocity, dtype=float),
        "fixed": fixed,
    }

#Function to create an RBC particle with the specified parameters
def make_rbc(radius, mass, position, velocity=None, fixed=False):
    return make_particle("RBC", radius, mass, position, velocity, fixed=fixed)


#Function to create a platelet particle with the specified parameters
def make_plt(radius, mass, position, velocity=None, fixed=False):
    return make_particle("PLT", radius, mass, position, velocity, fixed=fixed)


#Function to create a list of RBC particles
def make_rbc_population(radius, mass, positions, velocity=None, fixed=False):
    return [make_rbc(radius, mass, position, velocity, fixed) for position in positions]


#Function to create a list of platelet particles
def make_plt_population(radius, mass, positions, velocity=None, fixed=False):
    return [make_plt(radius, mass, position, velocity, fixed) for position in positions]


#Function to sample random positions while rejecting overlaps with existing particles
def sample_non_overlapping_positions(count,radius,x_bounds,y_bounds,rng,existing_particles=None,padding=0.0,max_attempts=1000,):
    existing_particles = [] if existing_particles is None else list(existing_particles)
    positions = []

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    attempts = 0

    while len(positions) < count:

        #If statemment to prevent infinite loops in case we can't find non-overlapping positions after n_attempts:
        if attempts >= max_attempts:
            raise ValueError("Could not place all particles without overlap.")

        candidate = np.array(
            [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)],
            dtype=float,
        )
        attempts += 1

        overlaps = False
        for other in existing_particles:
            min_distance = radius + other["radius"] + padding
            if np.linalg.norm(candidate - other["pos"]) < min_distance:
                overlaps = True
                break

        if overlaps:
            continue

        for other_position in positions:
            min_distance = 2 * radius + padding
            if np.linalg.norm(candidate - other_position) < min_distance:
                overlaps = True
                break

        if overlaps:
            continue

        positions.append(candidate)

    return [position.tolist() for position in positions]


#Function to create fixed RBC particles along the upper and lower vessel walls
def make_wall_rbc_particles(length, vessel_radius, radius, mass):
    x_positions = np.arange(radius, length)
    wall_y = vessel_radius - radius
    wall_positions = []

    for x_pos in x_positions:
        wall_positions.append([x_pos, wall_y])
        wall_positions.append([x_pos, -wall_y])

    return make_rbc_population(radius, mass, wall_positions, fixed=True)


#Function to calculate the plasma velocity profile across the vessel (Poiseuille flow)
def plasma_velocity_profile(y, vessel_radius, max_velocity):
    return max_velocity * (1 - (y / vessel_radius) ** 2)


#Function to calculate the drag force on a particle in the plasma flow (Stokes drag force)
def drag_force(particle, viscosity, vessel_radius, max_velocity):
    y = particle["pos"][1]
    u_fluid = np.array([plasma_velocity_profile(y, vessel_radius, max_velocity), 0.0])
    v_particle = particle["vel"]

    return 6 * np.pi * viscosity * particle["radius"] * (u_fluid - v_particle)

#Function to calculate the characteristic drag relaxation time for a particle
def drag_relaxation_time(particle, viscosity):
    return particle["mass"] / (6 * np.pi * viscosity * particle["radius"])

#Function to calculate the pairwise contact force between two particles
def pairwise_contact_force(particle, other_particle, spring_constant):
    displacement = particle["pos"] - other_particle["pos"]
    distance = np.linalg.norm(displacement)
    cutoff = particle["radius"] + other_particle["radius"]

    if distance == 0:
        direction = np.array([1.0, 0.0])
    else:
        direction = displacement / distance

    overlap = cutoff - distance
    if overlap <= 0:
        return np.zeros(2)

    return spring_constant * overlap * direction

#Function to calculate the total contact force on a particle from all others
def contact_force(particle, moving_particles, fixed_particles, spring_constant, self_index):
    total_force = np.zeros(2)

    #For loop: to calculate the contact force on a particle from all other moving particles:
    for index, other_particle in enumerate(moving_particles):
        if index == self_index:
            continue
        total_force += pairwise_contact_force(particle, other_particle, spring_constant)

    #for loop: to calculate the contact force on a particle from all fixed particles:
    for other_particle in fixed_particles:
        total_force += pairwise_contact_force(particle, other_particle, spring_constant)

    return total_force

#Function to update all particles with contact forces using a common snapshot
def update_particles_with_contact(particles,fixed_particles,dt,viscosity,vessel_radius,max_velocity,spring_constant,):
    particle_snapshots = [
        {
            **particle,
            "pos": particle["pos"].copy(),
            "vel": particle["vel"].copy(),
        }
        for particle in particles
    ]

    #for loop: to update all particles with contact forces using a common snapshot:
    for index, snapshot in enumerate(particle_snapshots):
        if snapshot["fixed"]:
            particles[index]["vel"] = np.zeros(2)
            particles[index]["pos"] = snapshot["pos"].copy()
            continue

        drag = drag_force(snapshot, viscosity, vessel_radius, max_velocity)
        contact = contact_force(snapshot, particle_snapshots, fixed_particles, spring_constant, index)
        total_force = drag + contact
        acceleration = total_force / snapshot["mass"]

        particles[index]["vel"] = snapshot["vel"] + acceleration * dt
        particles[index]["pos"] = snapshot["pos"] + particles[index]["vel"] * dt

    return particles
