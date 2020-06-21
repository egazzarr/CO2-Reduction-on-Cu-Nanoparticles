#!/usr/bin/env python
# coding: utf-8

# In[16]:


"""This code does the following:

- Plots the excess energies averaged over all simulations. (DeltaE = (E_{coh} - E_{pot})/{N^{2/3}})
- Reads a movie.xyz file and calculates the Pair Distribution Function (PDF) of the cluster 
  averaged over all time frames. It then individuates the first noticeable minimum, whihc corresponds
  to the radius for cut off. 
- Calculates the atop Generalized Coordination Number (aGCN) for each atom for each time frame.
  (aGCN_i=sum_j (CN_j/12)). 
- Plots histograms of relevant time steps for an easy comparison. 
- Outputs a new movie file (movie-aGCN.xyz) with a column (last columns) with the aGCN genome, readable 
  by Ovito and easily visualizable.
- Plots a heatmap of the aGCN occurrence over time (this takes some time). 
- Calculates the surface area (in cm^2) of the NP and its evolution through time and plots it
  ( Sum(from i to N) 4pir^2(1-(aGCN(i)/12)) ). 
- calculates the current density j (in mA/cm^2) for an applied potential between -1.299 and 0 V 
  from the relations by Zhao et al. (J. Phys. Chem. C 2016, 120, 28125-28130.) and through an equation 
  in which the constant is specific to each monometallic case. The equation for j can be 
  found in (J. Phys. Chem. B 2004, 108, 17886-17892). It then plots j at initial, 
  middle and final time steps.
- calculates the mass activity (MA) (in mA/mg) evolution through time of the NP through the 
  formula MA = (A_(surf)*SA) / M, where SA is the current density for a specific applied potential (-1.1V)
  and M is the NP's mass. 
  
"""

from asap3 import FullNeighborList
from ase import Atoms
from ase.io import read
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.interpolate import interp1d
import seaborn as sns
import pickle
import numpy as np
from itertools import groupby
from collections import namedtuple
from scipy.ndimage import gaussian_filter1d
import math 

"""YOUR folder, the code will automatically select the files needed. 
Make sure you have a movie.xyz file and an energy.out file. 
If not, comment some of the following lines to address your specific needs."""

folder= "..."   #insert your folder!
filename = str(folder)+'movie.xyz'
traj = read(filename, index = ':')

"""Check that the time in the energy file is in the fourth column. 
For growth, change [:,4] to [:,5]. """
energy= np.loadtxt(str(folder)+'energy.out')[:,4]
PATH_TO_NEW_MOVIE = str(folder)+"movie-AGCN.xyz"

"""Check that the time in the energy file is in the first column"""
tempo=np.loadtxt(str(folder)+'energy.out')[:, 0]
time_red=tempo/1000   #converting in to ns. 

"""CONSTANTS TO BE UPDATED AND INSERTED MANUALLY BY THE USER, depending on the 
monometallic species used and all its properties, such as mass and radius. """
sigma=2  #how much you want to smoothen out your functions?
applied_V=1.1  #at whihc voltage do you want the mass activity calculation to happen?
U_bins = 1299 #binning of the voltages (v)
beta=  1/((8.6173303E-5)*(300))   #Boltzmann constant at room temperature. Change the temperature at whihc you want the caatlytic activities to be performed at. 
r=1.28*10**(-9)  #atomic radius of your atoms (in cm)?
mass_cu = 1.0552e-19  #mass of your atoms (in mg)?
C=-30082651629632.555 # constant in equation, to be calculated from initial conditions
natoms = 586  #number of atoms in system?
mass_NP = mass_cu * natoms


"""Plots average excess energy evolution from simulations coming from different irands. 
Modify in a way that if you have more than four simulations, """
def plot_energies(filename):
    sigma=4                #decide how much you want to smoothen out the curve with a Gaussian filter
    E_tot=[]
    for j in range(4):
        E=np.loadtxt(str(folder)+'energy.out')[:, 4]
        E.tolist()
        E_tot.append(E)
    E_avg=(E_tot[0]+E_tot[1]+E_tot[2]+E_tot[3])/4
    plt.plot(time_red, gaussian_filter1d(E_avg, sigma))
    plt.show()
    plt.close()
    return

"""Calculates the distances between atoms."""
r_cut=10
def read_trajectory(filename, r_cut):
    atom_number_list = [atoms.get_atomic_numbers() for atoms in traj]
    flat_atom_number = np.concatenate(atom_number_list)
    elements = np.unique(flat_atom_number, return_counts=False)
    all_distances = []
    for i, atoms in enumerate(traj):
        atoms.set_cell([[100, 0, 0], [0, 100, 0], [0, 0, 100]])
        nl = FullNeighborList(r_cut, atoms=atoms)
        for i in np.arange(len(atoms)):
            indices, positions, distances = nl.get_neighbors(i)
            all_distances.extend(distances**0.5)
    return all_distances

"""Creates the r_cut_cn, the cut_off radius for further aGCN calculations. """
def create_function(distances, r_cut):
    y, bin_edges = np.histogram(distances, bins = np.linspace(0, r_cut, r_cut*100+1))
    x = bin_edges[:-1] + 0.005
    y = np.array(y) / sum(np.array(y))
    spline = interp1d(x, y)
    values = np.linspace(1, 5, 100)
    x=np.asarray(x)
    minima=[]
    """ Finding the minima in a very general way, finding the 10th y-value (percentage of 
    atoms within a certain radius) value before and after a point on the x-axis """
    for i in range (len(y)-10):
        if y[i]<y[i+10] and y[i]<y[i-10]:              
            minima.append(x[i])
    global r_cut_cn
    """r_cut_cn is the first encountered minima, used for cn calculation"""
    r_cut_cn=minima[0]                       
    print("Radius for cutoff=", r_cut_cn)
    plt.plot(values, spline(values))
    plt.show()
    return spline, r_cut_cn  

"""Atop Genralized Coordination Number calculation. Returns different arrays:agcn, a 2D array
(every 1D composing it refers to a single time step) of aGCNs between 2 and 9, agcn_tot_1D, a 1D array of all the values 
over all time frames, and agcn_tot_2D, a 2D array of all aGCN values (1D arrays correspond to singular
time frames). """
def cn_generator (filename, r_cut_cn):
    global agcn, agcn_tot_1D, agcn_tot_2D
    agcn_tot_1D=[]
    agcn_tot_2D=[]
    agcn=[] #containing arrays of aGCN at every time step

    for i, atoms in enumerate(traj):
        #frame by frame
        cn_frame=[]
        agcn_frame=[]
        agcn_tot_frame=[]
        atoms.set_cell([[100, 0, 0], [0, 100, 0], [0, 0, 100]])
        ind=[]
        for j in np.arange(len(atoms)):
            nl = FullNeighborList(r_cut_cn, atoms=atoms)
            indices, positions, distances = nl.get_neighbors(j)
            ind.append([int(k) for k in indices])
            distancej=[]    
            distancej.extend(distances**0.5)
            cnj=len(distancej)
            cn_frame.append(cnj)
        for l in np.arange(len(atoms)):
            cc=ind[l][:]
            list=[]
            for m in range(len(cc)):
                list.append(cn_frame[ind[l][m]])
                sm=sum(list)/12
            agcn_tot_1D.append(sm)
            agcn_tot_frame.append(sm)
                #we only want surface atoms, whihc have an atop GCN smaller than 11, from the freezing graphs
            if sm<=9:
                agcn_frame.append(sm)
        agcn_frame=np.asarray(agcn_frame)
        agcn_tot_frame=np.asarray(agcn_tot_frame)
        #rounding it to first decimal figure 
        agcn_frame=np.around(agcn_frame, 2)
        agcn_tot_frame=np.around(agcn_tot_frame, 2)
        agcn.append(agcn_frame)
        agcn_tot_2D.append(agcn_tot_frame)
    agcn=np.asarray(agcn)  #2D array: only surface atop GCN rounded at first decimal figure, each subarray corresponding to one time frame
    return (agcn, agcn_tot_1D, agcn_tot_2D)

def plot_hist (filename):

    plt.hist(agcn[0], bins=45, histtype= 'step', lw = 3, color='orange', label='liquid')
    plt.hist(agcn[int(len(agcn)/2)], bins=45,histtype= 'step', lw=2, color='purple', label = 'freezing T')
    plt.hist(agcn[len(agcn)-1], bins=45, histtype= 'step', lw=2, color = 'r', label = 'solid')
    plt.xlabel('aGCN', size='15')
    plt.ylabel('Distribution', size='15')
    plt.xticks(size='13')
    plt.yticks(size='13')
    #plt.legend(prop={'size': 13}, bbox_to_anchor=(1, 1))
    plt.savefig(str(folder)+'full_hist.png', bbox_inches='tight')
    plt.show()
    plt.close()  

    return 

def heatmap(filename):
    occ_long=[]
    for i in range(len(agcn)):
        (n, bins, patches)=plt.hist(agcn[i], bins=90, density=True)
        occ_long.append(n)
    occ_long=np.asarray(occ_long)
    global corrected_occ
    corrected_occ=np.zeros((90, len(agcn)))
    for i in range(90):
        corrected_occ[-i-1]=np.asarray(occ_long[:, i])
    print(corrected_occ)
    sns.heatmap(corrected_occ, vmax=1, xticklabels=False, yticklabels=False, cbar=False)
    plt.savefig(str(folder)+'occurrency.png', bbox_inches='tight', dpi=400)
    plt.show()
    plt.close()
    return(corrected_occ)
    
def readMovieFileXYZ(path_to_movie):
    """
    Reads a LoDiS movie.xyz file and fetches the coordinates
    for each atom for each frame.
    Input:
        path_to_movie: path to movie.xyz
    Returns:
        Named tuple read_movie:
            - read_movie.Frames: list of frames; each is an array of atoms each described by [Atom, x, y, z, Col]
            - read_movie.Headers: list of the movie frames headers"""
    read_file_chars = []
    with open(path_to_movie, 'r') as file:
        for line in file:
            read_file_chars.append(line)
    # 1. Delete line jump
    read_file_chars = [line[:-1] for line in read_file_chars]
    read_file_chars
    # 2. Separate line by line
    grouped_lines = [([list(group) for k, group in groupby(line,lambda x: x == " ") if not k]) for line in read_file_chars]
    # 3. Concatenate charaters
    joined_string = [[''.join(info_elem) for info_elem in grouped_line] for grouped_line in grouped_lines]
    # 4. Regroup into list of lists. Elements of outerlist are movie frames
    merged_frames = []
    current_frame = []
    for line in joined_string:
        if(line==joined_string[0]):
            if len(current_frame)!=0:
                merged_frames.append(current_frame)
            current_frame=[]
        else:
            current_frame.append(line)
    merged_frames.append(current_frame)
    # 5. Removing second line of header
    movie_headers_all_frames = [frame[0] for frame in merged_frames]
    merged_frames = [frame[1:] for frame in merged_frames]
    # 6. Converting coordinates and pressure to floats
    for frame in merged_frames:
        for line in frame:
            line[1] = float(line[1]) # x coord
            line[2] = float(line[2]) # y coord
            line[3] = float(line[3]) # z coord
    Movie = namedtuple('Movie', 'Frames Headers')
    read_movie = Movie(merged_frames, movie_headers_all_frames)
    return(read_movie)



def catalytic_analysis (filename):
    current=[]
    mass_activity=[]
    y=[]
    for j in range(0, len(agcn)):
        surf_area=0
        for i in range(len(agcn[j])):
            once = 4*np.pi*r**2*(1-agcn[j][i]/12)      
            surf_area = surf_area + once   
        y.append(surf_area)
        surface_area=gaussian_filter1d(y, sigma)
        site=np.zeros((U_bins))

        for h in range(U_bins):
            spec=0
            for m in range(len(corrected_occ)):
                if m<31:
                    sitecurrent = C*np.exp(((0.162 * m/10 - 1.11)-(h*0.001))*beta)*m/10*corrected_occ[m][j]/len(agcn[j])
                elif 31<=m<81:
                    sitecurrent = C*np.exp(((-0.067 * m/10 - 0.416)-(h*0.001))*beta)*m/10*corrected_occ[m][j]/len(agcn[j])
                else:
                    sitecurrent = C*np.exp(((-0.222 * m/10 + 0.849)-(h*0.001))*beta)*m/10*corrected_occ[m][j]/len(agcn[j])
                spec=spec+sitecurrent
            site[h]=spec
        current.append(site)
        mass_activity.append(-site[int(1299-applied_V*1000)]*surf_area/mass_NP)   
    
    """Calculation and plot of the surface area, the dashed line corresponds to the surface area per time
    step, while the solid line has been smoothened throuhg the Gaussian filter by whatever amount 
    you desire. Change sigma to explore. """
    plt.xlabel("Time(ns)")
    plt.ylabel("Surface area (cm^2)")
    plt.plot(time_red, y, color='black',linestyle='dashed')
    plt.plot(time_red, surface_area, color='black')
    plt.savefig(str(folder)+'surf_area.png', bbox_inches='tight', dpi=200)
    plt.show()
    plt.close()
    
    """Plotting the current density at different applied potentials for different time steps."""
    potentials = np.linspace(-1.299, 0, U_bins)
    plt.plot(potentials, current[int(len(traj)-1)], color='orange',lw=3, label='final')
    plt.plot(potentials, current[int(len(traj)/2)-1], color='purple', label='middle')
    plt.plot(potentials, current[0], color='r', label='initial')
    plt.xlabel('V vs RHE')
    plt.ylabel('j (mA/cm^2)')
    plt.savefig(str(folder)+'current densities.png', bbox_inches='tight', dpi=200)
    plt.show()
    plt.close()

    """Mass activity plots at desired applied_V."""
    plt.xlabel('Time(ns)')
    plt.ylabel('MA (mA/mg)')
    plt.plot(time_red, mass_activity, linestyle='dashed', color='black')
    plt.plot(time_red, gaussian_filter1d(mass_activity, sigma), linestyle='solid', color='black')
    plt.savefig(str(folder)+'mass_activity.png', bbox_inches='tight', dpi=200)
    plt.show()
    plt.close()
    
    return

if __name__ == '__main__':
    
    plot_energies(filename)
    
    tic = time.time()
    distances  = read_trajectory(filename, r_cut)
    toc = time.time()
    print("Time to calculate distances: %.2f [s]" %(toc-tic))

    tic = time.time()
    create_function(distances, r_cut)
    toc = time.time()
    print("Time to create spline: %.2f [s]" %(toc-tic))

    tic = time.time()
    agcn, agcn_tot_1D, agcn_tot_2D= cn_generator(filename, r_cut_cn)
    toc = time.time()
    print("Time to calculate aGCN: %.2f [s]" %(toc-tic))
    
    plot_hist(filename)
    
    heatmap(filename)
    
    catalytic_analysis(filename)
    
with open(PATH_TO_NEW_MOVIE, 'a+') as newmovie: # Mode chosen: append
    movie = readMovieFileXYZ(filename)
    NATOM = int(len(movie[0][0]))
    open(PATH_TO_NEW_MOVIE, 'w').close() #Clear old movie pressure file
    for frame_num, current_frame in enumerate(movie.Frames):
        with open(PATH_TO_NEW_MOVIE, 'a+') as newmovie: # Mode chosen: append
            num_lines = sum(1 for line in open(PATH_TO_NEW_MOVIE))
            if (num_lines==0): # No newline for first line -- bugs Ovito if there is newline at beginning
                newmovie.write(str(NATOM)+'\n')
            else:
                newmovie.write('\n' + str(NATOM)+'\n')
            newmovie.write('\t'.join(str(item) for item in movie.Headers[frame_num]))
            for atom_index, atom_info in enumerate(current_frame):
                print(atom_index, frame_num)
                atom_info[-1] = agcn_tot_2D[frame_num][atom_index] # Adding pressure to tuple
                newmovie.write('\n')
                newmovie.write('  \t'.join(str(item) for item in atom_info))



# In[ ]:




