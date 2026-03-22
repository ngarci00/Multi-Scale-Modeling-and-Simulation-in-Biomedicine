import numpy as np

#Function to create a generic particle dictionary with the specified parameters
def make_particle(kind, radius, mass, position, velocity=None, activated=False, fixed=False):
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
        "fixed": fixed,
    }

#Function to create an RBC particle with the specified parameters
def make_rbc(radius, mass, position, velocity=None, fixed=False):
    """Create one red blood cell particle."""
    return make_particle("RBC", radius, mass, position, velocity, fixed=fixed)

#Function to create a platelet particle with the specified parameters
def make_plt(radius, mass, position, velocity=None, activated=False):
    """Create one platelet particle."""
    return make_particle("PLT", radius, mass, position, velocity, activated)

#Function to create a list of RBC particles:
def make_rbc_population(radius, mass, positions, velocity=None, fixed=False):
    """Create a list of RBC particles from an iterable of positions."""
    return [make_rbc(radius, mass, position, velocity, fixed) for position in positions]

#Function to create a list of platelet particles:
def make_plt_population(radius, mass, positions, velocity=None, activated=False):
    """Create a list of platelet particles from an iterable of positions."""
    return [make_plt(radius, mass, position, velocity, activated) for position in positions]

#Function to sample random positions within the specified bounds:
def sample_random_positions(count, x_bounds, y_bounds, rng):
    """Sample uniformly random 2D positions inside the provided bounds."""
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    positions = []
    for _ in range(count):
        x_pos = rng.uniform(x_min, x_max)
        y_pos = rng.uniform(y_min, y_max)
        positions.append([x_pos, y_pos])

    return positions

#Function to sample random positions while rejecting overlaps with existing particles
def sample_non_overlapping_positions(count,radius,x_bounds,y_bounds,rng,existing_particles=None,padding=0.0,max_attempts=10000,):
    """Sample random positions while rejecting overlaps with existing particles."""
    existing_particles = [] if existing_particles is None else list(existing_particles)
    positions = []

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    attempts = 0

    #Keep sampling until we have enough non-overlapping positions or exceed max attempts
    while len(positions) < count:
        if attempts >= max_attempts:
            raise ValueError("Could not place all particles without overlap.")

        candidate = np.array(
            [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)],
            dtype=float,
        )
        attempts += 1

        overlaps = False
        #Check against existing particles for overlap
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
    """Create fixed RBC particles along the upper and lower vessel walls."""
    x_positions = np.arange(radius, length)
    wall_y = vessel_radius - radius
    wall_positions = []

    for x_pos in x_positions:
        wall_positions.append([x_pos, wall_y])
        wall_positions.append([x_pos, -wall_y])

    return make_rbc_population(radius, mass, wall_positions, fixed=True)

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

    return 6 * np.pi * viscosity * particle["radius"] * (u_fluid - v_particle) #Viscious drag force based on Stokes' law for a sphere in a fluid


def drag_relaxation_time(particle, viscosity):
    """Characteristic drag relaxation time m / (6*pi*mu*r) for one particle."""
    return particle["mass"] / (6 * np.pi * viscosity * particle["radius"])

#Function to calculate the pairwise contact force between two particles based on a repulsive spring model
def pairwise_contact_force(particle, other_particle, spring_constant):
    """Repulsive spring force if particles overlap, zero otherwise."""
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

#Function to calculate the total contact force on a particle from all other particles using the pairwise contact force
def contact_force(particle, moving_particles, fixed_particles, spring_constant, self_index):
    """Total contact force on a particle from all other particles using the pairwise contact force."""
    total_force = np.zeros(2)

    for index, other_particle in enumerate(moving_particles):
        if index == self_index:
            continue
        total_force += pairwise_contact_force(particle, other_particle, spring_constant)

    for other_particle in fixed_particles:
        total_force += pairwise_contact_force(particle, other_particle, spring_constant)

    return total_force

#Function to find the nearest point on the damaged wall:
def nearest_damage_point(position, damage_region):
    """Return the closest point on the damaged wall segment to a particle position."""
    x_coord = np.clip(position[0], damage_region["x_min"], damage_region["x_max"])
    return np.array([x_coord, damage_region["y"]], dtype=float)

#Function to check if a platelet is within the adhesion cutoff distance of the damaged region:
def platelet_in_damage_zone(particle, damage_region, adhesion_cutoff):
    """Return True if a platelet is close enough to the damage region to activate."""
    if particle["kind"] != "PLT":
        return False

    target = nearest_damage_point(particle["pos"], damage_region)
    return np.linalg.norm(particle["pos"] - target) <= adhesion_cutoff

#Function to calculate the adhesion force on an activated PLTs from the damaged wall and other PLTs:
def wall_adhesion_force(particle, damage_region, adhesion_spring, adhesion_cutoff):
    """Attractive force pulling an activated platelet toward the damage region."""
    if particle["kind"] != "PLT" or not particle["activated"]:
        return np.zeros(2)

    target = nearest_damage_point(particle["pos"], damage_region)
    displacement = particle["pos"] - target
    distance = np.linalg.norm(displacement)

    if distance == 0 or distance > adhesion_cutoff:
        return np.zeros(2)

    direction = displacement / distance
    return -adhesion_spring * (adhesion_cutoff - distance) * direction

#Function to calculate the adhesion force between activated PLTs based on their distance and the adhesion cutoff:
def platelet_pair_adhesion_force(particle, other_particle, adhesion_spring, adhesion_cutoff):
    """Attractive platelet-platelet adhesion force for activated platelets."""
    if particle["kind"] != "PLT" or other_particle["kind"] != "PLT":
        return np.zeros(2)
    if not particle["activated"] or not other_particle["activated"]:
        return np.zeros(2)

    displacement = particle["pos"] - other_particle["pos"]
    distance = np.linalg.norm(displacement)

    if distance == 0 or distance > adhesion_cutoff:
        return np.zeros(2)

    direction = displacement / distance
    return -adhesion_spring * (adhesion_cutoff - distance) * direction


def adhesion_force(particle,moving_particles,damage_region,adhesion_spring,adhesion_cutoff,self_index):
    """Total adhesion force on a platelet from the damage region and other activated platelets."""
    total_force = wall_adhesion_force(particle,damage_region,adhesion_spring,adhesion_cutoff) #Adhesion force from the damaged wall

    for index, other_particle in enumerate(moving_particles):
        if index == self_index:
            continue
        total_force += platelet_pair_adhesion_force(particle,other_particle,adhesion_spring,adhesion_cutoff) #Adhesion force from other activated platelets

    return total_force #Total force = wall adhesion + platelet-platelet adhesion

#Function to update a single particle's velocity and position based on drag force only 
def update_my_particle_drag(particle, dt, viscosity, vessel_radius, max_velocity):
    """Update the particle's velocity and position based on the drag force."""
    f_drag = drag_force(particle, viscosity, vessel_radius, max_velocity)
    acceleration = f_drag / particle["mass"]

    particle["vel"] = particle["vel"] + acceleration * dt
    particle["pos"] = particle["pos"] + particle["vel"] * dt

    return particle

#Function to update all particles with contact forces using a common snapshot to ensure consistent interactions
def update_particles_with_contact(particles,fixed_particles,dt,viscosity,vessel_radius,max_velocity,spring_constant,):
    """Advance all moving particles using drag plus contact from a common snapshot."""
    particle_snapshots = [
        {
            **particle, #state of particle at the start of the time step
            "pos": particle["pos"].copy(), 
            "vel": particle["vel"].copy(),
        }
        for particle in particles
    ]
    #Calculate forces and update each particle based on the snapshot to ensure consistent interactions
    for index, snapshot in enumerate(particle_snapshots):
        drag = drag_force(snapshot, viscosity, vessel_radius, max_velocity)
        contact = contact_force(snapshot,particle_snapshots,fixed_particles,spring_constant,index,)
        total_force = drag + contact
        acceleration = total_force / snapshot["mass"]

        particles[index]["vel"] = snapshot["vel"] + acceleration * dt
        particles[index]["pos"] = snapshot["pos"] + particles[index]["vel"] * dt

    return particles

#Function to update all particles with both contact and adhesion forces using a common snapshot
def update_particles_with_adhesion(particles,fixed_particles,dt,viscosity,vessel_radius,max_velocity,contact_spring,damage_region,adhesion_spring,adhesion_cutoff,):
    """Advance moving particles using drag, contact, and platelet-specific adhesion."""
    #snapshot of particle state at the start of the time step
    particle_snapshots = [
        {
            **particle,
            "pos": particle["pos"].copy(),
            "vel": particle["vel"].copy(),
        }
        for particle in particles
    ]
    #For loop: Check for PLT activation based on proximity to the damaged region
    for index, snapshot in enumerate(particle_snapshots):
        if platelet_in_damage_zone(snapshot, damage_region, adhesion_cutoff):
            snapshot["activated"] = True
            particles[index]["activated"] = True

    #Calculate forces and update each particle based on the snapshot 
    for index, snapshot in enumerate(particle_snapshots):
        drag = drag_force(snapshot, viscosity, vessel_radius, max_velocity)
        contact = contact_force(snapshot,particle_snapshots,fixed_particles,contact_spring,index)
        adhesion = adhesion_force(snapshot,particle_snapshots,damage_region,adhesion_spring,adhesion_cutoff,index,)
        total_force = drag + contact + adhesion
        acceleration = total_force / snapshot["mass"]

        particles[index]["vel"] = snapshot["vel"] + acceleration * dt
        particles[index]["pos"] = snapshot["pos"] + particles[index]["vel"] * dt

    return particles
