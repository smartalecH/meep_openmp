import meep as mp
from meep.materials import Al
import numpy as np

def main():
    lambda_min = 0.4         # minimum source wavelength
    lambda_max = 0.8         # maximum source wavelength
    fmin = 1/lambda_max      # minimum source frequency
    fmax = 1/lambda_min      # maximum source frequency
    fcen = 0.5*(fmin+fmax)   # source frequency center
    df = fmax-fmin           # source frequency width

    resolution = 55
    L = 10                   # length of OLED        
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
    sim.fields.step()
    sim.fields.reset_timers()

    for _ in range(100):
        sim.fields.step()

    sim.print_times()
    sim.output_times('oled_timing_statistics.csv')

if __name__ == '__main__':
    main()