import meep as mp
import numpy as np
import os
import argparse
import math
np.random.seed(0)

OMP_NUM_THREADS = os.getenv('OMP_NUM_THREADS')

def main(args):
    if args.dispersion:
        from meep.materials import cSi
    else:
        cSi = mp.Medium(index=3)
    
    wvl_min = 0.4
    wvl_max = 1.0
    fmin = 1/wvl_max
    fmax = 1/wvl_min
    fcen = 0.5*(fmin+fmax)
    df = fmax-fmin

    res = 50
    nperiods = 30        # number of lattice periods
    nfreq = 50  # number of frequency bins
    dpml = 2.0            # PML thickness
    dabs = 2.0            # Absorber thickness
    a = 0.5               # lattice period
    dfrac = 0.2           # maximum x/y positional disorder of rod as fraction of lattice period a        
    rt = a*0.2            # top lattice rod radius
    ht = 0.5              # top lattice rod height
    rb = a*0.5            # bottom lattice rod radius
    hb = 0.5              # bottom lattice rod height
    tair_top = 0.3        # top air thickness
    tcSi_top = 0.6        # top lattice thickness
    tgap = 0.2            # gap thickness
    tcSi_bot = 0.6        # bottom lattice thickness
    tair_bot = 0.3        # bottom air thickness

    sz = dpml+tair_top+tcSi_top+tgap+tcSi_bot+tair_bot+dabs
    sxy = nperiods*a

    cell_size = mp.Vector3(sxy,sxy,sz)

    boundary_layers = [mp.PML(thickness=dpml,direction=mp.Z,side=mp.High),
                       mp.Absorber(thickness=dabs,direction=mp.Z,side=mp.Low)]

    def rod_top(cx,cy):
        dpos = dfrac*a*0.2
        return mp.Cylinder(height=ht,
                           radius=rt,
                           material=mp.air,
                           center=mp.Vector3(cx+dpos,cy+dpos,0.5*sz-dpml-tair_top-0.5*ht))

    def rod_bot(cx,cy):
        dpos = dfrac*a*0.2
        return mp.Cylinder(height=hb,
                           radius=rb,
                           material=mp.air,
                           center=mp.Vector3(cx+dpos,cy+dpos,0.5*sz-dpml-tair_top-tcSi_top-tgap-0.5*hb))

    geometry = [mp.Block(material=cSi,
                         size=mp.Vector3(mp.inf,mp.inf,tcSi_top),
                         center=mp.Vector3(0,0,0.5*sz-dpml-tair_top-0.5*tcSi_top)),
                mp.Block(material=cSi,
                         size=mp.Vector3(mp.inf,mp.inf,tcSi_bot),
                         center=mp.Vector3(0,0,0.5*sz-dpml-tair_top-tcSi_top-tgap-0.5*tcSi_bot))]

    for cx in np.arange(-0.5*sxy+0.5*a,-0.5*sxy+0.5*a+nperiods*a,a):
        for cy in np.arange(-0.5*sxy+0.5*a,-0.5*sxy+0.5*a+nperiods*a,a):
            geometry.append(rod_top(cx,cy))
            geometry.append(rod_bot(cx,cy))

    # rotation angle of incident planewave; counter clockwise (CCW) about Y axis, 0 degrees along +X axis
    theta_in = math.radians(0)

    # k (in source medium) with correct length (plane of incidence: XY)
    k = mp.Vector3(z=fcen).rotate(mp.Vector3(y=1), theta_in)

    def pw_amp(k,x0):
        def _pw_amp(x):
            return cmath.exp(1j*2*math.pi*k.dot(x+x0))
        return _pw_amp

    src_pt = mp.Vector3(z=0.5*sz-dpml-0.1*tair_top)
    sources = [mp.Source(mp.GaussianSource(fcen,fwidth=df),
                         component=mp.Ex,
                         center=src_pt,
                         size=(sxy,sxy,0)
                         #amp_func=pw_amp(k,src_pt)
                         )]
    
    sim = mp.Simulation(resolution=res,
                        cell_size=cell_size,
                        boundary_layers=boundary_layers,
                        #k_point=k,
                        geometry=geometry,                    
                        sources=sources,
                        eps_averaging=False,
                        split_chunks_evenly=False,
                        collect_stats=True)

    #refl_pt = mp.Vector3(0,0,0.5*sz-dpml-0.5*tair_top)
    #refl_flux = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=refl_pt, size=mp.Vector3(sxy,sxy,0)))

    #tran_pt1 = mp.Vector3(0,0,0.5*sz-dpml-tair_top-tcSi_top-0.5*tgap)
    #tran_flux1 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=tran_pt1, size=mp.Vector3(sxy,sxy,0)))

    #tran_pt2 = mp.Vector3(0,0,-0.5*sz+dabs+0.5*tair_bot)
    #tran_flux2 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=tran_pt2, size=mp.Vector3(sxy,sxy,0)))
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
    sim.output_times('{}tandem_timing_statistics_{}_{}.csv'.format(args.fprefix,mp.count_processors(),OMP_NUM_THREADS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-res', type=int, default=120, help='resolution (default: 50 pixels/um)')
    parser.add_argument('-a_start', type=float, default=0.43, help='starting periodicity (default: 0.43 um)')
    parser.add_argument('-a_end', type=float, default=0.33, help='ending periodicity (default: 0.33 um)')
    parser.add_argument('-s_cav', type=float, default=0.146, help='cavity length (default: 0.146 um)')
    parser.add_argument('-rr', type=float, default=0.28, help='hole radius (default: 0.28 um)')
    parser.add_argument('-hh', type=float, default=0.22, help='waveguide height (default: 0.22 um)')
    parser.add_argument('-ww', type=float, default=0.50, help='waveguide width (default: 0.50 um)')
    parser.add_argument('-dabs', type=float, default=1.00, help='absorber thickness (default: 1.00 um)')
    parser.add_argument('-Ndef', type=int, default=3, help='number of defect periods (default: 3)')
    parser.add_argument('-Nwvg', type=int, default=20, help='number of waveguide periods (default: 8)')
    parser.add_argument('--dispersion', action='store_true', default=False, help='use dispersive materials? (default: False)')
    parser.add_argument('-f', '--fprefix', default='', help="File name for output file. Should end in .csv")
    args = parser.parse_args()
    main(args)
