import meep as mp
import numpy as np
import os
import argparse
import math

OMP_NUM_THREADS = os.getenv('OMP_NUM_THREADS')

def main(args):
    if args.dispersion:
        from meep.materials import Si, SiO2_aniso
    else:
        Si = mp.Medium(index=3)
        SiO2_aniso = mp.Medium(index=1.4)
    
    h = args.hh
    w = args.w
    a = args.a
    d = args.d
    N = args.nperiods
    N = N+1

    wvl_cen = 1.55
    dair_sub = 0.5

    dpml = wvl_cen
    boundary_layers = [mp.PML(dpml)]

    sxy = 2*(N+dpml)
    sz = dpml+dair_sub+h+dair_sub+dpml
    cell_size = mp.Vector3(sxy,sxy,sz)

    geometry = []

    # rings of Bragg grating
    for n in range(N,0,-1):
        geometry.append(mp.Cylinder(material=Si,
                                    center=mp.Vector3(0,0,0),
                                    radius=n*a,
                                    height=h))
        geometry.append(mp.Cylinder(material=mp.air,
                                    center=mp.Vector3(0,0,0),
                                    radius=n*a-d,
                                    height=h))

    # remove left half of Bragg grating rings to form semi circle
    geometry.append(mp.Block(material=mp.air,
                             center=mp.Vector3(-0.5*N*a,0,0),
                             size=mp.Vector3(N*a,2*N*a,h)))
    geometry.append(mp.Cylinder(material=Si,
                                center=mp.Vector3(0,0,0),
                                radius=a-d,
                                height=h))

    # angle sides of Bragg grating
    
    # rotation angle of sides relative to Y axis (degrees)
    rot_theta = -math.radians(args.rot_theta)
    
    pvec = mp.Vector3(0,0.5*w,0)
    cvec = mp.Vector3(-0.5*N*a,0.5*N*a+0.5*w,0)
    rvec = cvec-pvec
    rrvec = rvec.rotate(mp.Vector3(0,0,1), rot_theta)

    geometry.append(mp.Block(material=mp.air,
                             center=pvec+rrvec,
                             size=mp.Vector3(N*a,N*a,h),
                             e1=mp.Vector3(1,0,0).rotate(mp.Vector3(0,0,1),rot_theta),
                             e2=mp.Vector3(0,1,0).rotate(mp.Vector3(0,0,1),rot_theta),
                             e3=mp.Vector3(0,0,1)))

    pvec = mp.Vector3(0,-0.5*w,0)
    cvec = mp.Vector3(-0.5*N*a,-(0.5*N*a+0.5*w),0)
    rvec = cvec-pvec
    rrvec = rvec.rotate(mp.Vector3(0,0,1),-rot_theta)

    geometry.append(mp.Block(material=mp.air,
                             center=pvec+rrvec,
                             size=mp.Vector3(N*a,N*a,h),
                             e1=mp.Vector3(1,0,0).rotate(mp.Vector3(0,0,1),-rot_theta),
                             e2=mp.Vector3(0,1,0).rotate(mp.Vector3(0,0,1),-rot_theta),
                             e3=mp.Vector3(0,0,1)))
    
    # input waveguide
    geometry.append(mp.Block(material=mp.air,
                             center=mp.Vector3(-0.25*sxy,0.5*w+0.5*a,0),
                             size=mp.Vector3(0.5*sxy,a,h)))
    geometry.append(mp.Block(material=mp.air,
                             center=mp.Vector3(-0.25*sxy,-(0.5*w+0.5*a),0),
                             size=mp.Vector3(0.5*sxy,a,h)))
    geometry.append(mp.Block(material=Si,
                             center=mp.Vector3(-0.25*sxy,0,0),
                             size=mp.Vector3(0.5*sxy,w,h)))

    # substrate
    geometry.append(mp.Block(material=SiO2_aniso,
                             center=mp.Vector3(0,0,-0.5*sz+0.25*(sz-h)),
                             size=mp.Vector3(mp.inf,mp.inf,0.5*(sz-h))))

    # mode frequency
    fcen = 1/wvl_cen
    
    sources = [mp.EigenModeSource(src=mp.GaussianSource(fcen, fwidth=0.2*fcen),
                                  size=mp.Vector3(0,sxy-2*dpml,sz-2*dpml),
                                  center=mp.Vector3(-0.5*sxy+dpml+1.5,0,0),
                                  eig_match_freq=True,
                                  eig_parity=mp.ODD_Y,
                                  eig_kpoint=mp.Vector3(1.5,0,0),
                                  eig_resolution=32)]
    
    symmetries = [mp.Mirror(mp.Y,-1)]
    
    sim = mp.Simulation(resolution=args.res,
                        cell_size=cell_size,
                        boundary_layers=boundary_layers,
                        geometry=geometry,
                        sources=sources,
                        dimensions=3,
                        symmetries=symmetries,
                        split_chunks_evenly=False,
                        eps_averaging=False,
                        collect_stats=True)

    nearfield = sim.add_near2far(fcen, 0.2*fcen, args.nfreq, mp.Near2FarRegion(mp.Vector3(0,0,0.5*sz-dpml), size=mp.Vector3(sxy-2*dpml,sxy-2*dpml,0)))
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
    sim.output_times('{}grating_timing_statistics_{}_{}.csv'.format(args.fprefix,mp.count_processors(),OMP_NUM_THREADS))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hh', type=float, default=0.22, help='wavelength height (default: 0.22 um)')
    parser.add_argument('-w', type=float, default=0.50, help='wavelength width (default: 0.50 um)')
    parser.add_argument('-a', type=float, default=1.0, help='Bragg grating periodicity/lattice parameter (default: 1.0 um)')
    parser.add_argument('-d', type=float, default=0.5, help='Bragg grating thickness (default: 0.5 um)')
    parser.add_argument('-nperiods', type=int, default=5, help='number of grating periods')
    parser.add_argument('-rot_theta', type=float, default=20, help='rotation angle of sides relative to Y axis (default: 20 degrees)')
    parser.add_argument('-res', type=int, default=50, help='resolution (default: 30 pixels/um)')
    parser.add_argument('-nfreq', type=int, default=50, help='number of frequency bins (default: 50)')
    parser.add_argument('--dispersion', action='store_true', default=False, help='use dispersive materials? (default: False)')
    parser.add_argument('-f', '--fprefix', default='', help="File name for output file. Should end in .csv")
    args = parser.parse_args()
    main(args)
