#!/usr/bin/env python

import numpy as np
import argparse
import surfaces
from utilities import angstrom

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate vdw grid surface points')
    parser.add_argumnet('-i', '--xyz',                      dest='xyz', type=str, required=True, help='Name of xyz file to read coordinates and elements from')
    parser.add_argument('-r', '--surface-vdW-scale',        dest='surface_vdW_scale', default=2.0, type=float, help='Set the vdw radius scale parameter') 
    parser.add_argument('-d', '--surface-point-density',    dest='surface_point_density', default=20.0, type=float, help='Set the vdw surface point density') 
    parser.add_argument('-o',                               dest='output', type=str, default='out.grid')

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    elements = []
    with open(args.xyz, "r") as f:
        n_atoms = int(f.readline())
        coordinates = np.zeros((natoms,3), dtype=np.float64)
        comment = f.readline()
        for i in range(n_atoms):
            data = f.readline().split()
            elements.append(data[0])
            coordinates[i,:] = [float(x)*angstrom for x in data[1:4]]
    print(elements)
    print(coordinates) 
