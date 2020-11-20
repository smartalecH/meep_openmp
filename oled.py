import meep as mp
from meep.materials import Al
import numpy as np
import os
import argparse

OMP_NUM_THREADS = os.getenv('OMP_NUM_THREADS')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=float, default=10.0, help='design size')
    parser.add_argument('--fprefix',type=str, default="")
    args = parser.parse_args()

    lambda_min = 0.4         # minimum source wavelength
    lambda_max = 0.8         # maximum source wavelength
    fmin = 1/lambda_max      # minimum source frequency
    fmax = 1/lambda_min      # maximum source frequency
    fcen = 0.5*(fmin+fmax)   # source frequency center
    df = fmax-fmin           # source frequency width

    resolution = 55
    L = args.L               # length of OLED        
    nfreq = 100              # number of frequency bins
    tABS = 0.5               # absorber thickness
    tPML = 0.5               # PML thickness
    tGLS = 0.8               # glass thickness
    tORG = 0.4               # organic thickness
    tAl = 0.2                # aluminum thickness

    # length of computational cell along Z
    sz = tPML+tGLS+tORG+tAl+tPML
    # length of non-absorbing region of computational cell in X and Y
    sxy = L+2*tABS
    cell_size = mp.Vector3(sxy,sxy,sz)

    boundary_layers = [mp.Absorber(tABS,direction=mp.X),
                       mp.Absorber(tABS,direction=mp.Y),
                       mp.PML(tPML,direction=mp.Z)]

    geometry = [mp.Block(material=mp.Medium(index=1.5),
                         size=mp.Vector3(mp.inf,mp.inf,tPML+tGLS),
                         center=mp.Vector3(z=0.5*sz-0.5*(tPML+tGLS))),
                mp.Block(material=mp.Medium(index=1.75),
                         size=mp.Vector3(mp.inf,mp.inf,tORG),
                         center=mp.Vector3(z=0.5*sz-tPML-tGLS-0.5*tORG)),
                mp.Block(material=Al,
                         size=mp.Vector3(mp.inf,mp.inf,tAl),
                         center=mp.Vector3(z=0.5*sz-tPML-tGLS-tORG-0.5*tAl))]

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                         component=mp.Ez,
                         center=mp.Vector3(z=0.5*sz-tPML-tGLS-0.5*tORG))]

    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        boundary_layers=boundary_layers,
                        geometry=geometry,
                        sources=sources,
                        eps_averaging=False,
                        split_chunks_evenly=True,
                        collect_stats=True)

    srcbox_top = sim.add_flux(fcen, df, nfreq,
                              mp.FluxRegion(center=mp.Vector3(z=0.5*sz-tPML-tGLS), size=mp.Vector3(sxy,sxy,0), direction=mp.Z, weight=+1))
    srcbox_bot = sim.add_flux(fcen, df, nfreq,
                              mp.FluxRegion(center=mp.Vector3(z=0.5*sz-tPML-tGLS-0.8*tORG), size=mp.Vector3(sxy,sxy,0), direction=mp.Z, weight=-1))

    glass_top = sim.add_flux(fcen, df, nfreq,
                             mp.FluxRegion(center=mp.Vector3(z=0.5*sz-tPML), size = mp.Vector3(sxy,sxy,0), direction=mp.Z, weight=+1))

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
    sim.output_times('{}oled_timing_statistics_{}_{}.csv'.format(args.fprefix,mp.count_processors(),OMP_NUM_THREADS))

if __name__ == '__main__':
    main()
