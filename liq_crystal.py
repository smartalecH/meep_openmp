import meep as mp
import numpy as np
import os
import argparse
from meep.materials import cSi
import math
np.random.seed(0)

OMP_NUM_THREADS = os.getenv('OMP_NUM_THREADS')

def main(args):
    if args.dispersion:
        from meep.materials import SiO2_aniso
    else:
        SiO2_aniso = mp.Medium(index=1.4)
    dpml = 1.0             # PML thickness
    dsub = 1.0             # substrate thickness
    dpad = 1.0             # padding thickness
    dslab = 5.0            # slab thickness    
    ph_twisted = 70        # chiral layer twist angle for bilayer grating
    gp = 6.5               # grating period

    k_point = mp.Vector3(0,0,0)

    pml_layers = [mp.PML(thickness=dpml,direction=mp.X),
                  mp.PML(thickness=dpml,direction=mp.Z)]

    n_0 = 1.55
    delta_n = 0.159
    epsilon_diag = mp.Matrix(mp.Vector3(n_0**2,0,0),
                             mp.Vector3(0,n_0**2,0),
                             mp.Vector3(0,0,(n_0+delta_n)**2))

    wvl = 0.54             # center wavelength
    fcen = 1/wvl           # center frequency
    df = 0.2*fcen          # frequency width

    sx = dpml+dsub+args.dcry+args.dcry+dpad+dpml
    sy = gp
    sz = dpml+dslab+dpml

    cell_size = mp.Vector3(sx,sy,sz)

    # twist angle of nematic director; from equation 1b
    def phi(p):
        xx  = p.x-(-0.5*sx+dpml+dsub)
        if (xx >= 0) and (xx <= args.dcry):
            return math.pi*p.y/gp + ph_twisted*xx/args.dcry
        else:
            return math.pi*p.y/gp - ph_twisted*xx/args.dcry + 2*ph_twisted

    # return the anisotropic permittivity tensor for a uniaxial, twisted nematic liquid crystal
    def lc_mat(p):
        # rotation matrix for rotation around x axis
        Rx = mp.Matrix(mp.Vector3(1,0,0),mp.Vector3(0,math.cos(phi(p)),math.sin(phi(p))),mp.Vector3(0,-math.sin(phi(p)),math.cos(phi(p))))
        lc_epsilon = Rx * epsilon_diag * Rx.transpose()
        lc_epsilon_diag = mp.Vector3(lc_epsilon[0].x,lc_epsilon[1].y,lc_epsilon[2].z)
        lc_epsilon_offdiag = mp.Vector3(lc_epsilon[1].x,lc_epsilon[2].x,lc_epsilon[2].y)
        return mp.Medium(epsilon_diag=lc_epsilon_diag,epsilon_offdiag=lc_epsilon_offdiag)

    geometry = [mp.Block(center=mp.Vector3(-0.5*sx+0.5*(dpml+dsub)),
                         size=mp.Vector3(dpml+dsub,mp.inf,mp.inf),
                         material=SiO2_aniso),
                mp.Block(center=mp.Vector3(-0.5*sx+dpml+dsub+args.dcry),
                         size=mp.Vector3(2*args.dcry,mp.inf,mp.inf),
                         material=lc_mat(mp.Vector3()))]

    # linear-polarized planewave pulse source
    src_pt = mp.Vector3(-0.5*sx+dpml+0.3*dsub)
    sources = [mp.Source(mp.GaussianSource(fcen,fwidth=df),
                         component=mp.Ez,
                         center=src_pt,
                         size=mp.Vector3(0,sy,sz)),
               mp.Source(mp.GaussianSource(fcen,fwidth=df),
                         component=mp.Ey,
                         center=src_pt,
                         size=mp.Vector3(0,sy,sz))]

    sim = mp.Simulation(resolution=args.res,
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        k_point=k_point,
                        sources=sources,
                        geometry=geometry,
                        eps_averaging=False,
                        split_chunks_evenly=False,                        
                        collect_stats=True)

    refl_pt = mp.Vector3(-0.5*sx+dpml+0.5*dsub)
    refl_flux = sim.add_flux(fcen, df, args.nfreq, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0,sy,sz)))

    tran_pt = mp.Vector3(0.5*sx-dpml-0.5*dpad)
    tran_flux = sim.add_flux(fcen, df, args.nfreq, mp.FluxRegion(center=tran_pt, size=mp.Vector3(0,sy,sz)))
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
    sim.output_times('{}liq_timing_statistics_{}_{}.csv'.format(args.fprefix,mp.count_processors(),OMP_NUM_THREADS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-res', type=int, default=50, help='resolution (default: 50 pixels/um)')
    parser.add_argument('-dcry', type=float, default=5.0, help='liquid crystal thickness (default: 5.0)')
    parser.add_argument('-nfreq', type=int, default=21, help='number of frequency bins (default: 21)')
    parser.add_argument('--dispersion', action='store_true', default=False, help='use dispersive materials? (default: False)')
    parser.add_argument('-f', '--fprefix', default='', help="File name for output file. Should end in .csv")
    args = parser.parse_args()
    main(args)
