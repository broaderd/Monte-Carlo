import numpy as np  
import time
import sys
import os

#Function to generate random spin vectors
def get_spin_vecs(size=4**3):
    lo = np.repeat(np.array([0.0, 0.0, 0.0])[None, :], size, axis=0) 
    hi = np.repeat(np.array([1.0, 1.0, 1.0])[None, :], size, axis=0)
    x = np.random.uniform(low=lo, high=hi).T
    u = x[0]
    v = x[1]
    r = x[2]
    theta = 2 * np.pi * u
    phi = np.arccos(2*v - 1)
    length = 1.5*(r**(1/3))
    ct, cp = np.cos(theta), np.cos(phi)
    st, sp = np.sin(theta), np.sin(phi)
    return np.array([length * ct * sp, length * st * sp, length * cp]).T

#Function to compute the face-centered cubic (fcc) lattice
def fbcc(x, y, z):
    return 3 * (x*x + y*y + z*z) - 2 * (x*y + y*z + z*x)

#Function to compute the extended body-centered cubic (xbcc) lattice
def get_xbcc(x, y, z):
    return np.array([-x+y+z, x-y+z, x+y-z])

#Function to get nearest neighbors in the lattice
def get_nn():
    nearest = [] 
    for Nd in range(1, 9):
        N = Nd
        nvals = []
        for i in range(-N, N+1):
            for j in range(-N, N+1):
                for k in range(-N, N+1):
                    tup = (i, j, k)
                    num = fbcc(*tup)
                    if num == Nd:
                        nvals.append(tup)
        if nvals != []:
            nearest.append(nvals)
    return nearest

#Function to generate undisturbed lattice positions
def undisturbed(side=8):
    pos = np.zeros(shape=(side*side*side, 3))
    n = 0
    for n1 in range(side):
        for n2 in range(side):
            for n3 in range(side):
                pos[n] = n1*a1 + n2*a2 + n3*a3
                n += 1
    return pos

#Function to generate disturbed lattice positions
def disturbed(T, d):
    new_pos = np.zeros(shape=(side*side*side, 3))
    for i in range(side**3):
        for j in range(3):
            new_pos[i, j] = T[i, j] + np.random.uniform(-1*d, d)
    return new_pos

#Function to compute the Heisenberg energy of the lattice
def hs_energy(spins, Jey, T1, Tprime1):
    Energy = 0
    landau = 0
    for n1 in range(side):
        for n2 in range(side):
            for n3 in range(side):
                for idx, nvals in enumerate(nn_list):
                    part_energy = 0
                    end = len(nvals)//2                
                    for nval in nvals[:end]:
                        delta_n1 = (n1 + nval[0]) % side
                        delta_n2 = (n2 + nval[1]) % side
                        delta_n3 = (n3 + nval[2]) % side
                        deltaspin = spins[delta_n1, delta_n2, delta_n3]
                        dist = T1[delta_n1, delta_n2, delta_n3] - T1[n1, n2, n3]
                        dist_ij = Tprime1[delta_n1, delta_n2, delta_n3] - Tprime1[n1, n2, n3]
                        lst = [side, side-1, 0, 1]
                        if n1 in lst or n2 in lst or n3 in lst:
                            r_n = get_distance(dist)
                            r_ij = get_distance(dist_ij)
                        else:
                            r_n = np.linalg.norm(dist)
                            r_ij = np.linalg.norm(dist_ij)
                        del_rij = abs(r_ij - r_n)
                        factor = (1 - (del_rij / r_n))**3
                        part_energy += Jey[idx] * factor * np.dot(spins[n1, n2, n3], deltaspin)   
                    Energy += part_energy 
                si = np.linalg.norm(spins[n1, n2, n3])
                landau += -5117.64*(si**2) + 1747.08*(si**4) + 588.13*(si**6)
    Total = landau + Energy
    Total /= side**3
    return Total

#Function to compute the distance between lattice points
def get_distance(dist_vector):
    dists = np.linalg.norm([dist_vector + displacements[i] for i in range(displacements.shape[0])],axis=1)
    return np.amin(dists)

#Function to compute local energy at a lattice point
def local_energy(n1, n2, n3, spins, Jey, T1, Tprime1):
    Energy = 0
    for idx, nvals in enumerate(nn_list):
        part_energy = 0
        landau = 0
        for nval in nvals:
            delta_n1 = (n1 + nval[0]) % side
            delta_n2 = (n2 + nval[1]) % side
            delta_n3 = (n3 + nval[2]) % side
            deltaspin = spins[delta_n1, delta_n2, delta_n3]
            dist = T1[delta_n1, delta_n2, delta_n3] - T1[n1, n2, n3]
            dist_ij = Tprime1[delta_n1, delta_n2, delta_n3] - Tprime1[n1, n2, n3]
            lst = [side, side-1, 0, 1]
            if n1 in lst or n2 in lst or n3 in lst:
                r_n = get_distance(dist)
                r_ij = get_distance(dist_ij)
            else:
                r_n = np.linalg.norm(dist)
                r_ij = np.linalg.norm(dist_ij)
            del_rij = abs(r_ij - r_n)
            factor = (1 - (del_rij / r_n))**3
            part_energy += Jey[idx] * factor * np.dot(spins[n1, n2, n3], deltaspin)
        Energy += part_energy 
    si = np.linalg.norm(spins[n1, n2, n3])
    landau += -5117.64*(si**2) + 1747.08*(si**4) + 588.13*(si**6) 
    return Energy + landau

#Function to get all possible displacements for a lattice
def get_all_displacements(side, a1, a2, a3):
    ii = 0
    displacements = np.empty((27,3))
    for ix in range(-1, 2):
        for iy in range(-1, 2):
            for iz in range(-1, 2):
                displacements[ii] = side*(ix*a1 + iy*a2 + iz*a3)
                ii += 1
    return displacements

#Function to compute magnetization of the lattice
def magnetisation(spins):
    mag = spins.sum(axis=(0,1,2))
    divide = spins.shape[0]**3
    mag /= divide
    mag = mag * mag
    return mag.sum()

#Function to perform Monte Carlo simulation
def monte(spins, beta, Jey, T1, Tprime1):
    s_old = np.zeros(3)
    s_new = np.zeros(3)
    side = spins.shape[0]
    nacc = 0
    for n1 in range(side):
        for n2 in range(side):
            for n3 in range(side):
                E_old = local_energy(n1, n2, n3, spins, Jey, T1, Tprime1) 
                s_old[:] = spins[n1, n2, n3, :]
                s_new[:] = get_spin_vecs(1)
                spins[n1, n2, n3, :] = s_new[:]
                E_prime = local_energy(n1, n2, n3, spins, Jey, T1, Tprime1)
                cost = E_prime - E_old
                if E_prime < E_old:
                    E_old = E_prime
                    nacc += 1
                else:
                    r = np.random.uniform(0, 1)
                    w = np.exp(-1 * beta * cost)
                    if r <= w:
                        E_old = E_prime
                        nacc += 1
                    else:
                        spins[n1, n2, n3, :] = s_old
    return nacc / side**3  

#Function to perform simulation
def simu(T1, Tprime1, Temperatures, Jey, Nsweep=4000, Neq=2000, fsave=sys.stdout): 
    for temp in Temperatures:
        time1 = time.time()
        beta = 1/(temp)
        en_temp, en_temp_squared = 0.0, 0.0
        cv_en_temp, cv_en_temp_squared = 0.0, 0.0
        mag_temp, mag_temp_squared = 0.0, 0.0
        chi_temp, chi_temp_squared = 0.0, 0.0
        avg_acc1 = 0.0
        avg_acc2 = 0.0
        for sweep in range(Nsweep):
            acc = monte(spins, beta, Jey, T1, Tprime1)
            avg_acc1 += acc
            if sweep >= Neq:
                Energy = hs_energy(spins, Jey, T1, Tprime1) 
                Energy_cv = hs_energy(spins, Jey, T1, Tprime1) * (side**3)
                magnet = magnetisation(spins)
                magnet_chi = magnetisation(spins) * (side**6)
                avg_acc2 += acc
                en_temp += Energy
                en_temp_squared += Energy * Energy
                cv_en_temp += Energy_cv
                cv_en_temp_squared += Energy_cv * Energy_cv
                mag_temp += np.sqrt(magnet)
                mag_temp_squared += magnet
                chi_temp += np.sqrt(magnet_chi)
                chi_temp_squared += magnet_chi
        en_temp /= (Nsweep - Neq)
        en_temp_squared /= (Nsweep - Neq)
        cv_en_temp /= (Nsweep - Neq)
        cv_en_temp_squared /= (Nsweep - Neq)
        Cv = (1/(temp**2)) * (cv_en_temp_squared - cv_en_temp**2)
        Cv /= side**3
        mag_temp /= (Nsweep - Neq)
        mag_temp_squared /= (Nsweep - Neq)
        chi_temp /= (Nsweep - Neq)
        chi_temp_squared /= (Nsweep - Neq)
        Chi = (1/temp) * (chi_temp_squared - chi_temp**2)
        avg_acc1 /= Nsweep
        avg_acc2 /= (Nsweep - Neq)
        avg_rej = 1 - avg_acc1 
        time2 = time.time()
        fsave.write(f"{temp:15.10f} {en_temp:15.10f} {mag_temp:15.10f} {Cv:15.10f} {Chi:15.10f} {avg_acc1:15.10f} {avg_rej:15.10f}\n")
        fsave.flush()
        print(f'Temperature = {temp:15.10f}, time = {(time2 - time1):15.10f}')
        spinsfilename = "8_{}K".format(temp)
        np.savez(os.path.join(folderpath, spinsfilename), spins, Tprime1, en_temp, cv_en_temp)
    return None

if __name__ == "__main__":
    a1 = np.array([-0.5, 0.5, 0.5])
    a2 = np.array([0.5, -0.5, 0.5])
    a3 = np.array([0.5, 0.5, -0.5])
    side = 8
    d = 0.01
    displacements = get_all_displacements(side, a1, a2, a3)
    np.random.seed(42)
    T = undisturbed(side)
    T1 = undisturbed(side).reshape(side, side, side, 3)
    Tprime = disturbed(T, d)
    Tprime1 = disturbed(T, d).reshape(side, side, side, 3)
    nn_list = get_nn()
    spins = get_spin_vecs(size=side*side*side).reshape(side, side, side, 3)
    Jey = np.array([-261.02, -169.37, -60.32], dtype=float)
    Energy = hs_energy(spins, Jey, T1, Tprime1)
    MaxT, MinT = 2000, 1e-6
    Temperatures = np.linspace(MaxT, MinT, 40)
    folderpath = "folder_path"
    fsave = open('location_to_save_file', 'w')
    _ = simu(T1, Tprime1, Temperatures, Jey, Nsweep=1000, Neq=500, fsave=fsave)
    fsave.close()
