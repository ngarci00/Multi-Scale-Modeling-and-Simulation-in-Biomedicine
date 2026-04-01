import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

#Geometry and variables
L = 400
D = 100 
dt =  0.1  
r_rbc = 2.2
r_plt = 1.1
N_rbc = 12
N_plt = 40
inlet_xmin = 5
inlet_xmax = 45
inlet_ymin = 10
inlet_ymax = 90
damaged_start = 0.4*L  
damaged_end = 0.6*L

dt = 0.2     
frames_total = 2000  
skip_frames = 5      
attraction_force = 1.7  
umax = 1     

activated_plt = np.zeros(N_plt, dtype=int)  # 0: mobile, 1: fixed in clot
clot_color = 'orange'  # accumulated clot color
activation_radius = 20  # distance to activate

k_a = 0.05          # adhesion force
r_a = 4 * r_plt     # adhesion radius

# Generate initial positions 
def generate_positions(N, r, max_retries=1000):
    positions = []
    inlet_area = (inlet_xmax - inlet_xmin) * (inlet_ymax - inlet_ymin)
    if N * np.pi * r**2 > 0.6 * inlet_area:  # only can occupy 60%
        print(f"Warning: High packing for r={r}, N={N}. Reducing density.")
        N = int(0.5 * inlet_area / (np.pi * r**2))  # reduce number of cells if it occupies over 60%
        print(f"N adjusted to {N}")
    
    # puts cell in a random position
    for _ in range(N):
        for retry in range(max_retries):
            x = np.random.uniform(inlet_xmin, inlet_xmax)
            y = np.random.uniform(inlet_ymin, inlet_ymax)
            new_pos = np.array([x, y])
            overlap = any(np.linalg.norm(new_pos - p) < 2*r for p in positions)
            if not overlap:
                positions.append(new_pos)
                break
        else:
            # If overlap, changes the position
            if positions:
                dists = [np.linalg.norm(new_pos - p) for p in positions]
                new_pos = positions[np.argmin(dists)] + np.random.normal(0, 3*r, 2)
            positions.append(new_pos)
            print(f"Fallback used for particle {_}")
    return np.array(positions)


pos_rbc = generate_positions(N_rbc, r_rbc)
pos_plt = generate_positions(N_plt, r_plt)
vel_rbc = np.zeros((N_rbc, 2))   # velocity vectors
vel_plt = np.zeros((N_plt, 2))

# Poiseuille function
def poiseuille_flow(y):
    return umax * (1 - (2*y/D - 1)**2)

def update(frame): 
    ax.clear()
    
    #plot of the vessel with the damaged area
    x = np.linspace(0, L, 1000) 
    ax.plot(x, np.zeros_like(x), 'b-', linewidth=8, label='Lower wall')
    ax.plot(x, D * np.ones_like(x), 'b-', linewidth=8, label='Upper wall')
    ax.plot(np.linspace(damaged_start, damaged_end, 100), np.zeros(100), 'k-', linewidth=8, label='Damaged region')
    
    for _ in range(skip_frames):
        # behaviour of RBC
        for i in range(N_rbc):
            y = pos_rbc[i,1]
            vx = poiseuille_flow(y)
            vy = 0
            pos_rbc[i,0] += vx * dt
            pos_rbc[i,1] += vy * dt
            if pos_rbc[i,0] > L: #if it goes out of the vessel through the right, enters again in random position on the left
                pos_rbc[i,0] = inlet_xmin
                pos_rbc[i,1] = np.random.uniform(inlet_ymin, inlet_ymax)
        
        # behavior of platelets
        for i in range(N_plt):
            if activated_plt[i] == 1:
                continue
            
            y = pos_plt[i,1]
            vx = poiseuille_flow(y)
            vy = 0

            dx = pos_plt[i,0] - (damaged_start + damaged_end)/2   # Vector components to calculate distance to damaged region
            dy = pos_plt[i,1] - 0
            dist_damaged = np.sqrt(dx**2 + dy**2)
            
            if dist_damaged < 40:
                if dist_damaged > 1e-6:
                    vx += -attraction_force * 0.5 * (dx / dist_damaged) # If distance is small, apply stronger attraction
                    vy += -attraction_force * (dy / dist_damaged)
                
                if dist_damaged < activation_radius:                # Sticking if the distance is very small
                    activated_plt[i] = 1
                    pos_plt[i,1] = 2 * r_plt
            
            # Movement
            pos_plt[i,0] += vx * dt         
            pos_plt[i,1] += vy * dt

            # Bounce, control bounds (the cells can't go past the vessel walls)
            if pos_plt[i,1] <= r_plt:
                pos_plt[i,1] = r_plt
            elif pos_plt[i,1] >= D - r_plt:
                pos_plt[i,1] = D - r_plt
            
            if pos_plt[i,0] > L:
                pos_plt[i,0] = inlet_xmin
                pos_plt[i,1] = np.random.uniform(inlet_ymin, inlet_ymax)

        # Every platelet
        for i in range(N_plt):
            # Compared with every other platelet
            for j in range(i+1, N_plt):
                # Calculate distance
                dx = pos_plt[i,0] - pos_plt[j,0]
                dy = pos_plt[i,1] - pos_plt[j,1]
                dist = np.sqrt(dx**2 + dy**2)

                r_c = 2 * r_plt  # minimum distance between two platelets center

                if dist < r_c and dist > 1e-6:
                    overlap = r_c - dist

                    nx = dx / dist
                    ny = dy / dist

                    # If overlap, move the platelets
                    correction = 0.5 * overlap

                    pos_plt[i,0] += correction * nx
                    pos_plt[i,1] += correction * ny

                    pos_plt[j,0] -= correction * nx
                    pos_plt[j,1] -= correction * ny

                    # Keep inside the vessels bounds
                    pos_plt[i,1] = np.clip(pos_plt[i,1], r_plt, D - r_plt)
                    pos_plt[j,1] = np.clip(pos_plt[j,1], r_plt, D - r_plt)
        # same for RBC
        for i in range(N_rbc):
            for j in range(i+1, N_rbc):
                dx = pos_rbc[i,0] - pos_rbc[j,0]
                dy = pos_rbc[i,1] - pos_rbc[j,1]
                dist = np.sqrt(dx**2 + dy**2)

                r_c = 2 * r_rbc

                if dist < r_c and dist > 1e-6:
                    overlap = r_c - dist

                    nx = dx / dist
                    ny = dy / dist

                    correction = 0.5 * overlap

                    pos_rbc[i,0] += correction * nx
                    pos_rbc[i,1] += correction * ny

                    pos_rbc[j,0] -= correction * nx
                    pos_rbc[j,1] -= correction * ny
                    pos_rbc[i,1] = np.clip(pos_rbc[i,1], r_rbc, D - r_rbc)
                    pos_rbc[j,1] = np.clip(pos_rbc[j,1], r_rbc, D - r_rbc)

        # same between RBC and platelets
        for i in range(N_rbc):
            for j in range(N_plt):
                dx = pos_rbc[i,0] - pos_plt[j,0]
                dy = pos_rbc[i,1] - pos_plt[j,1]
                dist = np.sqrt(dx**2 + dy**2)

                r_c = r_rbc + r_plt

                if dist < r_c and dist > 1e-6:
                    overlap = r_c - dist

                    nx = dx / dist
                    ny = dy / dist

                    correction = 0.5 * overlap

                    pos_rbc[i,0] += correction * nx
                    pos_rbc[i,1] += correction * ny
                    pos_plt[j,0] -= correction * nx
                    pos_plt[j,1] -= correction * ny

                    pos_rbc[i,1] = np.clip(pos_rbc[i,1], r_rbc, D - r_rbc)
                    pos_plt[j,1] = np.clip(pos_plt[j,1], r_plt, D - r_plt)
                    
        # adhesion of the platelets on the damage area and between them in the damaged area
        for i in range(N_plt):
            if activated_plt[i] != 1:
                continue
            for j in range(i+1, N_plt):
                if activated_plt[j] != 1:
                    continue

                dx = pos_plt[i,0] - pos_plt[j,0]
                dy = pos_plt[i,1] - pos_plt[j,1]
                dist = np.sqrt(dx**2 + dy**2)

                if dist > 1e-6 and dist < r_a:
                    nx = dx / dist
                    ny = dy / dist

                    r_c = 2 * r_plt

                    if dist > r_c:
                        # force proportional to extension
                        force_mag = k_a * (dist - r_c)

                        # Convert force to displacement
                        correction = 0.5 * force_mag * dt

                        pos_plt[i,0] -= correction * nx
                        pos_plt[i,1] -= correction * ny
                        pos_plt[j,0] += correction * nx
                        pos_plt[j,1] += correction * ny

    # Plot all the cells
    for i, pos in enumerate(pos_plt):
        color = clot_color if activated_plt[i] == 1 else 'yellow'
        alpha = 0.9 if activated_plt[i] == 1 else 0.8
        ax.add_patch(Circle(pos, r_plt, color=color, alpha=alpha))
    
    for pos in pos_rbc:
        ax.add_patch(Circle(pos, r_rbc, color='red', alpha=0.7))
    
    ax.set_xlim(0, L)
    ax.set_ylim(-10, D+10)
    ax.set_aspect('equal')
    ax.set_title(f'Frame {frame} - Clot: {np.sum(activated_plt)} platelets')
    ax.legend()

# Animation video
fig, ax = plt.subplots(figsize=(12, 4))
ani = FuncAnimation(fig, update, frames=500, interval=50, blit=False)
ani.save('simulation.mp4', writer='ffmpeg', dpi=150)
print("Animation saved as 'simulation.mp4' - open with VLC.")