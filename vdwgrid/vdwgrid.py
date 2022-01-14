#!/usr/bin/env python

import numpy as np
import argparse
from surfaces import compute_vdW_surface
from utilities import angstrom, symbol2number

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate vdw grid surface points')
    parser.add_argument('-i', '--xyz',                      dest='xyz', type=str, required=True, help='Name of xyz file to read coordinates and elements from')
    parser.add_argument('-r', '--surface-vdW-scale',        dest='surface_vdW_scale', default=2.0, type=float, help='Set the vdw radius scale parameter') 
    parser.add_argument('-d', '--surface-point-density',    dest='surface_point_density', default=20.0, type=float, help='Set the vdw surface point density') 
    parser.add_argument('-o',                               dest='output', type=str, default='out.grid')

    args = parser.parse_args()
    elements = []
    with open(args.xyz, "r") as f:
        n_atoms = int(f.readline())
        coordinates = np.zeros((n_atoms,3), dtype=np.float64)
        comment = f.readline()
        for i in range(n_atoms):
            data = f.readline().split()
            elements.append(data[0])
            coordinates[i,:] = np.array([float(x)*angstrom for x in data[1:4]])
    atomic_charges = np.array([symbol2number[element] for element in elements])
    surface = compute_vdW_surface(atomic_charges, coordinates, args.surface_point_density, args.surface_vdW_scale)
    with open(args.output, "w") as f:
        f.write("{}\n".format(surface.shape[0]))
        for point in surface:
            f.write("{:16.10f}  {:16.10f}  {:16.10f}\n".format(*point))
