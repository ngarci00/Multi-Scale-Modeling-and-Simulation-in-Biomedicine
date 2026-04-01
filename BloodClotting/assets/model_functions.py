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
        "adhered": False,
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


# Function to calculate the plasma velocity profile across the vessel (Poiseuille flow)
def plasma_velocity_profile(y, vessel_radius, max_velocity):
    return max_velocity * (1 - (y / vessel_radius) ** 2)


# Function to calculate the drag force on a particle in the plasma flow (Stokes drag force)
def drag_force(particle, viscosity, vessel_radius, max_velocity):
    y = particle["pos"][1]
    u_fluid = np.array([plasma_velocity_profile(y, vessel_radius, max_velocity), 0.0])
    v_particle = particle["vel"]

    return 6 * np.pi * viscosity * particle["radius"] * (u_fluid - v_particle)


# Once a platelet activates, damp its existing motion instead of continuing to
# drive it with the background plasma flow.
def activated_platelet_damping_force(particle, viscosity):
    return -6 * np.pi * viscosity * particle["radius"] * particle["vel"]


# Compute a stable timestep from the current particle state for the kinematic
# update used in this project.
def compute_stable_dt(
    particles,
    vessel_radius,
    max_velocity,
    contact_spring,
    damage_region,
    adhesion_spring,
    dt_max,
    displacement_fraction=0.25,
):
    min_dt = dt_max

    particle_snapshots = [
        {
            **particle,
            "pos": particle["pos"].copy(),
            "vel": particle["vel"].copy(),
        }
        for particle in particles
    ]

    for index, particle in enumerate(particle_snapshots):
        if particle["fixed"] or particle["adhered"]:
            continue

        base_velocity = np.array(
            [plasma_velocity_profile(particle["pos"][1], vessel_radius, max_velocity), 0.0]
        )
        contact = contact_force(particle, particle_snapshots, contact_spring, index)
        adhesion = adhesion_velocity(particle, particle_snapshots, damage_region, adhesion_spring)

        speed = np.linalg.norm(base_velocity + contact + adhesion)
        if speed <= 1e-12:
            continue

        particle_dt = displacement_fraction * particle["radius"] / speed
        min_dt = min(min_dt, particle_dt, dt_max)

    return max(min_dt, 1e-12)


# Function to calculate the nearest point on the damaged wall segment
def nearest_damage_point(position, damage_region):
    x_coord = np.clip(position[0], damage_region["x_min"], damage_region["x_max"])
    y_coord = damage_region.get("contact_y", damage_region["y"])
    return np.array([x_coord, y_coord], dtype=float)


# Build platelet-sized adhesion sites along the damaged wall patch.
def damage_target_sites(damage_region, platelet_radius):
    x_min = damage_region["x_min"] + platelet_radius
    x_max = damage_region["x_max"] - platelet_radius
    y_coord = damage_region.get("contact_y", damage_region["y"])

    if x_max <= x_min:
        return [np.array([(damage_region["x_min"] + damage_region["x_max"]) / 2, y_coord], dtype=float)]

    spacing = 2 * platelet_radius
    x_values = np.arange(x_min, x_max + 0.5 * spacing, spacing)
    return [np.array([x_value, y_coord], dtype=float) for x_value in x_values]


# Pick the nearest unoccupied adhesion site on the damaged wall.
def nearest_available_damage_target(particle, particles, damage_region):
    target_sites = damage_target_sites(damage_region, particle["radius"])
    occupied_targets = [other["pos"] for other in particles if other["kind"] == "PLT" and other["adhered"]]

    available_sites = []
    for site in target_sites:
        occupied = any(np.linalg.norm(site - occupied_site) < particle["radius"] for occupied_site in occupied_targets)
        if not occupied:
            available_sites.append(site)

    candidate_sites = available_sites if available_sites else target_sites
    distances = [np.linalg.norm(particle["pos"] - site) for site in candidate_sites]
    return candidate_sites[int(np.argmin(distances))].copy()


# Function to calculate the distance between a particle and the damaged wall segment

def distance_to_damage_region(particle, damage_region):
    target = nearest_damage_point(particle["pos"], damage_region)
    return np.linalg.norm(particle["pos"] - target)


# Function to check if a platelet is within the activation threshold of the damaged wall
def is_near_damage_region(particle, damage_region, threshold):
    if particle["kind"] != "PLT":
        return False

    return distance_to_damage_region(particle, damage_region) <= threshold


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

        if is_near_damage_region(snapshot, damage_region, threshold):
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


# Function to calculate the total particle-particle contact force on a moving particle
def contact_force(particle, moving_particles, spring_constant, self_index):
    total_force = np.zeros(2)

    for index, other_particle in enumerate(moving_particles):
        if index == self_index:
            continue
        total_force += pairwise_contact_force(particle, other_particle, spring_constant)

    return total_force


# Apply analytic upper/lower vessel walls, matching the simpler wall handling
# used in p3_bueno.py.
def apply_vessel_wall_bounds(particle, vessel_radius):
    if particle["fixed"] or particle["adhered"]:
        return

    y_min = -vessel_radius + particle["radius"]
    y_max = vessel_radius - particle["radius"]

    if particle["pos"][1] <= y_min:
        particle["pos"][1] = y_min
        particle["vel"][1] = abs(particle["vel"][1])
    elif particle["pos"][1] >= y_max:
        particle["pos"][1] = y_max
        particle["vel"][1] = -abs(particle["vel"][1])


# Function to calculate the wall adhesion force for activated platelets near the damaged region
def wall_adhesion_force(particle, damage_region, adhesion_cutoff, adhesion_strength):
    if particle["kind"] != "PLT" or not particle["activated"]:
        return np.zeros(2)

    target = nearest_damage_point(particle["pos"], damage_region)
    displacement = target - particle["pos"]
    distance = np.linalg.norm(displacement)

    if distance == 0:
        return np.zeros(2)

    # Once a platelet has activated, keep pulling it toward the damaged wall
    # target until it reaches and adheres there. Using a persistent spring-like
    # pull avoids the failure mode where the platelet activates near the patch,
    # drifts slightly past the cutoff, and then resumes its original path.
    return adhesion_strength * displacement


# Convert the activated platelet attraction into a direct velocity correction,
# following the kinematic update style used in p3_bueno.py.
def adhesion_velocity(particle, particles, damage_region, adhesion_strength):
    if particle["kind"] != "PLT" or not particle["activated"]:
        return np.zeros(2)

    target = nearest_available_damage_target(particle, particles, damage_region)
    displacement = target - particle["pos"]
    distance = np.linalg.norm(displacement)

    if distance <= 1e-12:
        return np.zeros(2)

    direction = displacement / distance

    # Bias the activated platelet motion strongly toward the wall-normal
    # direction so it does not keep drifting past the damaged region.
    x_velocity = 0.1 * adhesion_strength * direction[0]
    y_velocity = 3.0 * adhesion_strength * direction[1]
    return np.array([x_velocity, y_velocity])


# Stop an activated platelet once it has effectively reached the damaged site.
def adhere_platelet_at_target(particle, particles, damage_region):
    if particle["kind"] != "PLT" or not particle["activated"] or particle["adhered"]:
        return

    target = nearest_available_damage_target(particle, particles, damage_region)
    distance = np.linalg.norm(particle["pos"] - target)

    if distance <= particle["radius"]:
        particle["adhered"] = True
        particle["pos"] = target.copy()
        particle["vel"] = np.zeros(2)

# Function to update all particles with activation, drag, contact, and optional wall adhesion
def update_particles_with_activation_and_adhesion(
    particles,
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
        if snapshot["fixed"] or snapshot["adhered"]:
            particles[index]["vel"] = np.zeros(2)
            particles[index]["pos"] = snapshot["pos"].copy()
            continue

        if snapshot["kind"] == "PLT" and snapshot["activated"]:
            base_velocity = np.zeros(2)
        else:
            base_velocity = np.array(
                [plasma_velocity_profile(snapshot["pos"][1], vessel_radius, max_velocity), 0.0]
            )
        contact = contact_force(snapshot, particle_snapshots, contact_spring, index)
        adhesion = adhesion_velocity(
            snapshot,
            particle_snapshots,
            damage_region,
            k_adhesion,
        )
        velocity = base_velocity + contact + adhesion

        particles[index]["vel"] = velocity
        particles[index]["pos"] = snapshot["pos"] + velocity * dt
        apply_vessel_wall_bounds(particles[index], vessel_radius)
        adhere_platelet_at_target(particles[index], particles, damage_region)

    return particles
