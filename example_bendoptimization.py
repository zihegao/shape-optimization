# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:08:52 2025

@author: jbd0043
"""

import tidy3d as td
import tidy3d.plugins.adjoint as tda
from tidy3d.plugins.adjoint.web import run_local as run
from tidy3d.plugins.adjoint.utils.penalty import RadiusPenalty
from tidy3d.plugins.mode import ModeSolver

import numpy as np
import matplotlib.pylab as plt

import jax
import jax.numpy as jnp

wavelength = 1.5 #nm
freq0 = td.C_0/wavelength 

fwidth = freq0 / 10
num_freqs = 10
freqs = np.linspace(freq0 - fwidth/2, freq0 + fwidth/2, num_freqs)

#number of vertices
num_pts = 60
# angles associated with each vertex
# excludes 0 and pi/2
angles = np.linspace(0, np.pi/2, num_pts + 2)[1:-1]

# refractive indices of waveguide and subtrate (air above)
n_wg = 2.0
n_sub = 1.5

# min space between waveguide and PML
spc = 1 * wavelength

# length of input and output straight waveguide sections
t = 1 * wavelength

mode_spc = t / 2.0

# height of waveguide code
h = 0.7

# minimum, starting, and maximum allowed width of the waveguide
wmin = 0.5
wmid = 1.5 #starting
wmax = 2.5

# average radius of curvature of the bend
radius  = 6

# minimum allowed radius of curvature of the polygon
min_radius = 150e-3

# name of the monitor measuring the transmission amplitudes for optimization
monitor_name = 'mode'

# how many grid points per wavelength in the waveguide core material
min_steps_per_wvl = 30

# how many mode outputs to view 
num_modes = 3
mode_spec = td.ModeSpec(num_modes=num_modes)

# Define total simulation space (in 3D)
Lx = Ly = t + radius + abs(wmax - wmin) + spc
Lz = spc + h + spc

#%% map the -inf to inf optimization parameter to the minium and maximum thickness
# must be smooth, using hyperbolic tangent
def map_param_to_wg_width(param: float) -> float:
    param_01 = (jnp.tanh(param) + 1.0) / 2.0 # tanh centered at 0.5
    return wmax * param_01 + wmin * (1 - param_01)

#%% create the vertices of the waveguide at each given parameter position
def make_wg_vertices(params: np.ndarray) -> list:
    vertices = []
    # 2 vertices for input overlapping region
    vertices = []
    vertices.append((-Lx/2 + 1e-2, -Ly/2 + t + radius))
    vertices.append((-Lx/2 + t, -Ly/2 + t + radius + wmid/2))
    for angle, param in zip(angles, params):
        width_i = map_param_to_wg_width(param)
        radius_i = radius + width_i/2.0
        x = radius_i * np.sin(angle) -Lx/2 + t
        y = radius_i * np.cos(angle) -Ly/2 + t
        vertices.append((x, y))
    # 3 vertices for output overlapping region
    vertices.append((-Lx/2 + t + radius + wmid/2, -Ly/2 + t))
    vertices.append((-Lx/2 + t + radius, -Ly/2 + 1e-2))
    vertices.append((-Lx/2 + t + radius - wmid/2, -Ly/2 + t))
    for angle, param in zip(angles[::-1], params[::-1]):
        width_i = map_param_to_wg_width(param)
        radius_i = radius - width_i/2.0
        x = radius_i * np.sin(angle) -Lx/2 + t
        y = radius_i * np.cos(angle) -Ly/2 + t
        vertices.append((x, y))
    # final vertex for input overlapping region
    # order matters to create polyhedron correctly
    vertices.append((-Lx/2 + t, -Ly/2 + t + radius - wmid/2))
    return vertices

#%% generate and plot the vertices plot
params = np.zeros(num_pts)
vertices = make_wg_vertices(params)

plt.scatter(*np.array(vertices).T)
ax = plt.gca() # gets current axes
ax.set_aspect('equal')

#%% makes a polyhedron based on the vertices generated
def make_wg_polyhedron(params: np.ndarray) -> tda.JaxPolySlab:
    vertices = make_wg_vertices(params)
    return tda.JaxPolySlab(
        vertices = vertices,
        slab_bounds = (0, h),
        axis = 2,
        )

#%% plot cross section of the polyhedron 
polyhedron = make_wg_polyhedron(params)
ax = polyhedron.plot(z=0) #plot the cross section at z = 0

#%% assigns permittivity to the polyhedron (JaxStructure)
def make_wg_structure(params) -> tda.JaxStructure:
    polyhedron = make_wg_polyhedron(params)
    medium = tda.JaxMedium(permittivity=n_wg**2)
    return tda.JaxStructure(
        geometry=polyhedron,
        medium=medium)

#%% define polyhedron and material parameters of the static portions of the 90 degree bend
wg_in_polyhedron = td.Box.from_bounds(
    rmin=(-Lx/2-1,-Ly/2+t+radius-wmid/2,0),
    rmax=(-Lx/2+t+1e-3,-Ly/2+t+radius+wmid/2,h)
    )

wg_out_polyhedron = td.Box.from_bounds(
    rmin=(-Lx/2 + t + radius - wmid/2, -Ly/2 - 1, 0),
    rmax=(-Lx/2 + t + radius + wmid/2, -Ly/2 + t, h),
    )

substrate_polyhedron = td.Box.from_bounds(
    rmin=(-td.inf, -td.inf, -10000), #does not support infinite z extent
    rmax=(+td.inf, +td.inf, 0),
)

wg_in = td.Structure(geometry=wg_in_polyhedron,medium=td.Medium(permittivity=n_wg**2))
wg_out = td.Structure(geometry=wg_out_polyhedron,medium=td.Medium(permittivity=n_wg**2))
substrate = td.Structure(geometry=substrate_polyhedron,medium=td.Medium(permittivity=n_sub**2))


#%% Create a penalty function for bending
penalty = RadiusPenalty(min_radius=min_radius, alpha=1.0, kappa=10.0) #explain

def eval_penalty(params):
    vertices = make_wg_vertices(params)
    _vertices = jnp.array(vertices)
    vertices_top = _vertices[1:num_pts+3] # vertices on outer edge
    vertices_bot = _vertices[num_pts+4:] # certices on inner edge
    penalty_top = penalty.evaluate(vertices_top)
    penalty_bot = penalty.evaluate(vertices_bot)
    return (penalty_top + penalty_bot) / 2.0 # average the inner and outer curvature

#test
print(eval_penalty(params))

#%% Create Mode Source and Monitors
mode_width = wmid + 2 *spc
mode_height = Lz

#creates a range of frequencies to find the desired mode
mode_src = td.ModeSource(
    size=(0,mode_width,mode_height),
    center=(-Lx/2+t/2,-Ly/2+t+radius,0),
    direction='+',
    source_time=td.GaussianPulse(
        freq0=freq0,
        fwidth=fwidth
        )
    )

#monitors the central frequency for optimization
mode_mnt = td.ModeMonitor(
    size=(mode_width,0,mode_height),
    center=(-Lx/2+t+radius,-Ly/2+t/2,0),
    name=monitor_name,
    freqs=[freq0],
    mode_spec=mode_spec
    )

flux_mnt = td.FluxMonitor(
    size=(mode_width, 0, mode_height),
    center=(-Lx/2 + t + radius, -Ly/2 + t/2, 0),
    name="flux",
    freqs=[freq0],
    )

#monitors the entire transmitted frequency range
mode_mnt_bb = td.ModeMonitor(
    size=(mode_width, 0, mode_height),
    center=(-Lx/2+t+radius,-Ly/2+t/2,0),
    name='mode_bb',
    freqs=freqs.tolist(),
    mode_spec=mode_spec
    )

fld_mnt = td.FieldMonitor(
    size=(td.inf,td.inf,0),
    freqs=[freq0],
    name='field'
    )

#%% create simulation object and run mode solver
def make_sim(params, use_fld_mnt: bool = True) -> tda.JaxSimulation:
    monitors = [mode_mnt_bb, flux_mnt]
    if use_fld_mnt:
        monitors += [fld_mnt]
    input_structures = make_wg_structure(params)
    return tda.JaxSimulation(
        size=(Lx, Ly, Lz),
        input_structures=[input_structures],
        structures=[substrate, wg_in, wg_out],
        sources=[mode_src],
        output_monitors=[mode_mnt],
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        monitors=monitors,
        run_time = 10/fwidth,
        )

#test
bend_wg = make_wg_structure(params)
test = td.Simulation(
    size=(Lx, Ly, Lz),
    structures=[bend_wg,substrate, wg_in, wg_out],
    sources=[mode_src],
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl),
    boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
    monitors=[mode_mnt,mode_mnt_bb,flux_mnt,fld_mnt],
    run_time = 10/fwidth
    )
sim = make_sim(params)
f, (ax1, ax2) = plt.subplots(1,2,tight_layout=True,figsize=(10,4))
ax = sim.plot(z=0.01,ax=ax1)
ax = sim.plot(x=-Lx/2+t/2,ax=ax2)

td.config.logging_level = 'ERROR'

#%% Evalute the unoptimized Eigenmodes and select mode_convert mode source to single mode source of desired mode index

ms = ModeSolver(simulation=test,plane=mode_src,mode_spec=mode_spec,freqs=mode_mnt.freqs)
data = ms.solve()

print('Effective index of computed modes:', np.array(data.n_eff))

fig,axs = plt.subplots(num_modes,3,figsize=(14,10),tight_layout=True)
for mode_ind in range(num_modes):
    for field_ind, field_name in enumerate(('Ex','Ey','Ez')):
        field = data.field_components[field_name].sel(mode_index=mode_ind)
        ax = axs[mode_ind,field_ind]
        field.real.plot(x='y',y='z',ax=ax,cmap='RdBu')
        ax.set_title(f'{field_name}, mode_index={mode_ind}')

mode_index = 0
mode_src = ms.to_source(mode_index=mode_index,source_time=mode_src.source_time,direction=mode_src.direction)
        
# %% Define the objective function and test it by taking the gradient
def objective(params, use_fld_mnt:bool = True):
    sim = make_sim(params, use_fld_mnt = use_fld_mnt)
    sim_data = run(sim, task_name='bend', verbose=False)
    amps = sim_data[monitor_name].amps.sel(direction='-',mode_index=mode_index).values 
    #^ selects the complex amplitude associated with mode monitor
    #The mode monitor was initialized to only monitor the fundamental mode
    transmission = jnp.abs(jnp.array(amps))**2
    J = jnp.sum(transmission) - eval_penalty(params)
    return J, sim_data

val_grad = jax.value_and_grad(objective,has_aux=True) #computes gradient using Adjoint method
#
(val, sim_data), grad = val_grad(params)
print(f"Objective Function's Value = {val}")
print(f"Objective Function's gradient = {grad}")

# %% Optimization process
import optax
import numpy as np
import time

start_time = time.time()

iter_steps = 40
learning_rate = 0.1 #?

params = np.array(params).copy()
optimizer = optax.adam(learning_rate=learning_rate)
opt_state = optimizer.init(params)

# store history
objective_history = []
param_history = [params]
data_history = []

for i in range(iter_steps):
    # compute gradient and current objective function value
    (value, sim_data), gradient = val_grad(params)
    
    # multiply all by -1 to maximize obj_fn (instead of minimize)
    gradient = -np.array(gradient.copy())
    
    # outputs
    print(f'step = {i+1}')
    print(f'\tJ = {value:.4e}')
    print(f'\tgrad_norm = {np.linalg.norm(gradient):.4e}')
    
    updates, opt_state = optimizer.update(gradient,opt_state,params)
    params = optax.apply_updates(params,updates)
    
    # save history
    objective_history.append(value)
    param_history.append(params)
    data_history.append(sim_data)
    

end_time = time.time()
print(f'Optimization time: {(end_time - start_time)/3600.0:.3} hours')

fig = plt.plot(objective_history)
plt.xlabel('iteration number')
plt.ylabel('objective function: Transmission')
plt.title('optimization progress')

#%% plot comparison of intial and optimized geometry
sim_start = make_sim(param_history[0])
data_start = data_history[0]

sim_final = make_sim(param_history[-1])
data_final = data_history[-1]

f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,tight_layout=True,figsize=(10,6))

data_start.plot_field(field_monitor_name='field',field_name='E',val='abs^2',ax=ax1)
sim_start.plot(z=0,ax=ax2)
data_final.plot_field(field_monitor_name='field',field_name='E',val='abs^2',ax=ax3)
sim_final.plot(z=0,ax=ax4)
    
#%% animate the field pattern evolution over the entire optimization:
import matplotlib.animation as animation

fig, (ax1, ax2) = plt.subplots(1,2,tight_layout=False,figsize=(8,4))

def animate(i):
    sim_data_i = data_history[i]
    
    sim_i = sim_data_i.simulation.to_simulation()[0]
    sim_i.plot_eps(z=0, monitor_alpha=0.0,source_alpha=0.0, ax=ax1)
    
    sim_data_i.plot_field(field_monitor_name='field',field_name='E',val='abs^2',ax=ax2)
    fig.suptitle(f'iteration = {i}')
ani = animation.FuncAnimation(fig,animate,frames=len(data_history),interval=750)

writer = animation.FFMpegWriter(fps=1,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save('90degree_bend_wg_optimization.mp4', writer=writer)