import meep as mp
from meep.materials import Al
import numpy as np
import os
import argparse

OMP_NUM_THREADS = os.getenv('OMP_NUM_THREADS')

def main(args):
    resolution = args.res         # pixels/um

    a_start = args.a_start        # starting periodicity
    a_end = args.a_end            # ending periodicity
    s_cav = args.s_cav            # cavity length
    r = args.rr                   # hole radius (units of a)
    h = args.hh                   # waveguide height
    w = args.ww                   # waveguide width

    dair = 1.00                   # air padding
    dpml = 1.00                   # PML thickness

    Ndef = args.Ndef              # number of defect periods
    a_taper = mp.interpolate(Ndef, [a_start, a_end])
    dgap = a_end-2*r*a_end

    Nwvg = args.Nwvg              # number of waveguide periods
    sx = 2*(Nwvg*a_start+sum(a_taper))-dgap+s_cav
    sy = dpml+dair+w+dair+dpml
    sz = dpml+dair+h+dair+dpml

    cell_size = mp.Vector3(sx, sy, sz)
    boundary_layers = [mp.Absorber(args.dabs,direction=mp.X),
                       mp.PML(dpml,direction=mp.Y),
                       mp.PML(dpml,direction=mp.Z)]

    geometry = [mp.Block(material=Si,
                         center=mp.Vector3(),
                         size=mp.Vector3(mp.inf,w,h))]

    for mm in range(Nwvg):
        geometry.append(mp.Cylinder(material=mp.air,
                                    center=mp.Vector3(-0.5*sx+0.5*a_start+mm*a_start),
                                    radius=r*a_start,
                                    height=mp.inf))
        geometry.append(mp.Cylinder(material=mp.air,
                                    center=mp.Vector3(+0.5*sx-0.5*a_start-mm*a_start),
                                    radius=r*a_start,
                                    height=mp.inf))

    for mm in range(Ndef+2):
        geometry.append(mp.Cylinder(material=mp.air,
                                    center=mp.Vector3(-0.5*sx+Nwvg*a_start+(sum(a_taper[:mm]) if mm>0 else 0)+0.5*a_taper[mm]),
                                    radius=r*a_taper[mm],
                                    height=mp.inf))
        geometry.append(mp.Cylinder(material=mp.air,
                                    center=mp.Vector3(+0.5*sx-Nwvg*a_start-(sum(a_taper[:mm]) if mm>0 else 0)-0.5*a_taper[mm]),
                                    radius=r*a_taper[mm],
                                    height=mp.inf))

    lambda_min = 1.46        # minimum source wavelength
    lambda_max = 1.66        # maximum source wavelength
    fmin = 1/lambda_max
    fmax = 1/lambda_min
    fcen = 0.5*(fmin+fmax)
    df = fmax-fmin

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                         component=mp.Ey,
                         center=mp.Vector3())]

    symmetries = [mp.Mirror(mp.X,+1),
                  mp.Mirror(mp.Y,-1),
                  mp.Mirror(mp.Z,+1)]

    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        boundary_layers=boundary_layers,
                        geometry=geometry,
                        sources=sources,
                        dimensions=3,
                        symmetries=symmetries,
                        collect_stats=True,
                        eps_averaging=False,
                        split_chunks_evenly=False)
    mp.verbosity(2)
    sim.init_sim()
    mem = sim.get_estimated_memory_usage()
    npixels = sim.fragment_stats.num_pixels_in_box
    print("Memory usage: ",mem)
    print("Number of pixels: ",npixels)
    
    sim.fields.step()
    sim.fields.reset_timers()

    for _ in range(10):
        sim.fields.step()

    sim.print_times()
    sim.output_times('{}nanobeam_timing_statistics_{}_{}.csv'.format(args.fprefix,mp.count_processors(),OMP_NUM_THREADS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-res', type=int, default=50, help='resolution (default: 50 pixels/um)')
    parser.add_argument('-a_start', type=float, default=0.43, help='starting periodicity (default: 0.43 um)')
    parser.add_argument('-a_end', type=float, default=0.33, help='ending periodicity (default: 0.33 um)')
    parser.add_argument('-s_cav', type=float, default=0.146, help='cavity length (default: 0.146 um)')
    parser.add_argument('-rr', type=float, default=0.28, help='hole radius (default: 0.28 um)')
    parser.add_argument('-hh', type=float, default=0.22, help='waveguide height (default: 0.22 um)')
    parser.add_argument('-ww', type=float, default=0.50, help='waveguide width (default: 0.50 um)')
    parser.add_argument('-dabs', type=float, default=1.00, help='absorber thickness (default: 1.00 um)')
    parser.add_argument('-Ndef', type=int, default=3, help='number of defect periods (default: 3)')
    parser.add_argument('-Nwvg', type=int, default=8, help='number of waveguide periods (default: 8)')
    parser.add_argument('-f', '--fprefix', default='', help="File name for output file. Should end in .csv")
    args = parser.parse_args()
    main(args)
