# Nicolas Garcia Callejas - BENG 535 - Project 3: Blood Clotting
# This script serves as a collection of all the used functions needed in the main script to run the model.
import numpy as np


# Function to create a generic particle dictionary with the specified parameters

def make_particle(
    kind,
    radius,
    mass,
    position,
    velocity=None,
    fixed=False,
    activated=False,
    activation_time=0.0,
):
    if velocity is None:
        velocity = [0.0, 0.0]

    return {
        "kind": kind,
        "radius": radius,
        "mass": mass,
        "pos": np.array(position, dtype=float),
        "vel": np.array(velocity, dtype=float),
        "fixed": fixed,
        "activated": activated,
        "activation_time": activation_time,
    }


# Function to create an RBC particle with the specified parameters
def make_rbc(radius, mass, position, velocity=None, fixed=False):
    return make_particle("RBC", radius, mass, position, velocity, fixed=fixed)


# Function to create a platelet particle with the specified parameters
def make_plt(radius, mass, position, velocity=None, fixed=False, activated=False, activation_time=0.0):
    return make_particle(
        "PLT",
        radius,
        mass,
        position,
        velocity,
        fixed=fixed,
        activated=activated,
        activation_time=activation_time,
    )


# Function to create a list of RBC particles
def make_rbc_population(radius, mass, positions, velocity=None, fixed=False):
    return [make_rbc(radius, mass, position, velocity, fixed) for position in positions]


# Function to create a list of platelet particles
def make_plt_population(radius, mass, positions, velocity=None, fixed=False, activated=False, activation_time=0.0):
    return [
        make_plt(radius, mass, position, velocity, fixed, activated, activation_time)
        for position in positions
    ]


# Function to sample random positions while rejecting overlaps with existing particles
def sample_non_overlapping_positions(count, radius, x_bounds, y_bounds, rng, existing_particles=None, padding=0.0, max_attempts=1000):
    existing_particles = [] if existing_particles is None else list(existing_particles)
    positions = []

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    attempts = 0

    while len(positions) < count:
        if attempts >= max_attempts:
            raise ValueError("Could not place all particles without overlap.")

        candidate = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
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


# Function to create fixed RBC particles along the upper and lower vessel walls

def make_wall_rbc_particles(length, vessel_radius, radius, mass):
    x_positions = np.arange(radius, length)
    wall_y = vessel_radius - radius
    wall_positions = []

    for x_pos in x_positions:
        wall_positions.append([x_pos, wall_y])
        wall_positions.append([x_pos, -wall_y])

    return make_rbc_population(radius, mass, wall_positions, fixed=True)


# Function to calculate the plasma velocity profile across the vessel (Poiseuille flow)
def plasma_velocity_profile(y, vessel_radius, max_velocity):
    return max_velocity * (1 - (y / vessel_radius) ** 2)


# Function to calculate the drag force on a particle in the plasma flow (Stokes drag force)
def drag_force(particle, viscosity, vessel_radius, max_velocity):
    y = particle["pos"][1]
    u_fluid = np.array([plasma_velocity_profile(y, vessel_radius, max_velocity), 0.0])
    v_particle = particle["vel"]

    return 6 * np.pi * viscosity * particle["radius"] * (u_fluid - v_particle)


# Function to calculate the nearest point on the damaged wall segment
def nearest_damage_point(position, damage_region):
    x_coord = np.clip(position[0], damage_region["x_min"], damage_region["x_max"])
    y_coord = damage_region.get("contact_y", damage_region["y"])
    return np.array([x_coord, y_coord], dtype=float)


# Function to calculate the distance between a particle and the damaged wall segment

def distance_to_damage_region(particle, damage_region):
    target = nearest_damage_point(particle["pos"], damage_region)
    return np.linalg.norm(particle["pos"] - target)


# Function to check if a platelet is within the activation threshold of the damaged wall
def is_near_damage_region(particle, damage_region, threshold):
    if particle["kind"] != "PLT":
        return False

    return distance_to_damage_region(particle, damage_region) <= threshold


# Function to check if a platelet is near any activated platelet
def is_near_activated_platelet(particle, particles, self_index, threshold):
    if particle["kind"] != "PLT":
        return False

    for index, other_particle in enumerate(particles):
        if index == self_index:
            continue
        if other_particle["kind"] != "PLT" or not other_particle["activated"]:
            continue

        distance = np.linalg.norm(particle["pos"] - other_particle["pos"])
        if distance <= threshold:
            return True

    return False


# Function to update platelet activation using continuous dwell time
def update_platelet_activation(
    particles,
    damage_region,
    dt,
    threshold,
    activation_time_required,
):
    particle_snapshots = [
        {
            **particle,
            "pos": particle["pos"].copy(),
            "vel": particle["vel"].copy(),
        }
        for particle in particles
    ]

    for index, snapshot in enumerate(particle_snapshots):
        if snapshot["kind"] != "PLT":
            continue

        if snapshot["activated"]:
            particles[index]["activation_time"] = max(
                particles[index]["activation_time"], activation_time_required
            )
            continue

        near_damage = is_near_damage_region(snapshot, damage_region, threshold)
        near_activated_platelet = is_near_activated_platelet(
            snapshot,
            particle_snapshots,
            index,
            threshold,
        )

        if near_damage or near_activated_platelet:
            particles[index]["activation_time"] += dt
        else:
            particles[index]["activation_time"] = 0.0

        if particles[index]["activation_time"] >= activation_time_required:
            particles[index]["activated"] = True
            particles[index]["activation_time"] = activation_time_required

    return particles


# Function to calculate the pairwise contact force between two particles
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


# Function to calculate the total contact force on a particle from all others
def contact_force(particle, moving_particles, fixed_particles, spring_constant, self_index):
    total_force = np.zeros(2)

    for index, other_particle in enumerate(moving_particles):
        if index == self_index:
            continue
        total_force += pairwise_contact_force(particle, other_particle, spring_constant)

    for other_particle in fixed_particles:
        total_force += pairwise_contact_force(particle, other_particle, spring_constant)

    return total_force


# Function to calculate the wall adhesion force for activated platelets near the damaged region
def wall_adhesion_force(particle, damage_region, adhesion_cutoff, adhesion_strength):
    if particle["kind"] != "PLT" or not particle["activated"]:
        return np.zeros(2)

    target = nearest_damage_point(particle["pos"], damage_region)
    displacement = particle["pos"] - target
    distance = np.linalg.norm(displacement)

    if distance == 0 or distance > adhesion_cutoff:
        return np.zeros(2)

    direction = displacement / distance
    return -adhesion_strength * (adhesion_cutoff - distance) * direction


# Function to update all particles with activation, drag, contact, and optional wall adhesion
def update_particles_with_activation_and_adhesion(
    particles,
    fixed_particles,
    dt,
    viscosity,
    vessel_radius,
    max_velocity,
    contact_spring,
    damage_region,
    threshold,
    activation_time_required,
    adhesion_cutoff,
    k_adhesion,
):
    update_platelet_activation(
        particles,
        damage_region,
        dt,
        threshold,
        activation_time_required,
    )

    particle_snapshots = [
        {
            **particle,
            "pos": particle["pos"].copy(),
            "vel": particle["vel"].copy(),
        }
        for particle in particles
    ]

    for index, snapshot in enumerate(particle_snapshots):
        if snapshot["fixed"]:
            particles[index]["vel"] = np.zeros(2)
            particles[index]["pos"] = snapshot["pos"].copy()
            continue

        drag = drag_force(snapshot, viscosity, vessel_radius, max_velocity)
        contact = contact_force(snapshot, particle_snapshots, fixed_particles, contact_spring, index)
        adhesion = wall_adhesion_force(
            snapshot,
            damage_region,
            adhesion_cutoff,
            k_adhesion,
        )

        total_force = drag + contact + adhesion
        acceleration = total_force / snapshot["mass"]

        particles[index]["vel"] = snapshot["vel"] + acceleration * dt
        particles[index]["pos"] = snapshot["pos"] + particles[index]["vel"] * dt

    return particles
