'''
radiation integrals c/o Sergey Tomin
IBS c/o Ilya Agapov
'''


from scipy.integrate import simps
from ocelot.common.globals import *
from ocelot.cpbd.optics import trace_z, twiss
from ocelot.cpbd.beam import *
from ocelot.cpbd.elements import *
from ocelot.rad.undulator_params import *


def I2_ID(L, h0):
    return L/2.*h0*h0


def I3_ID(L, h0):
    return 4.*L/(3*pi)*h0**3


def I4_ID(L, h0, lu):
    return -3.*L*lu*lu*h0**4/(32*pi)


def I5_ID(L, h0, lu, beta_xc, Dx0, Dxp0):
    # it is the same as I5_exact and I5_exact2
    # beta_c - beta_x at the center of ID
    # Dx0, Dxp0 - Dx and Dxp at the beginning of ID
    nn = int(L/lu)
    I = ((h0**3 *L)/(108000 *pi**5 *beta_xc) *(144000* pi**4* (Dx0**2 + beta_xc**2 *Dxp0**2) +
        13500* (-1)**nn* h0* pi**3* Dx0* lu**2 + 15656 *h0**2* lu**4 +
        15* (-76 + 225* (-1)**nn) *h0**2 *pi* lu**4 +
        150 *h0* pi**2* lu**2* (480* Dx0 + h0* (4* L**2 + 48 *beta_xc**2 - lu**2))))
    return I

def radiation_integrals(lattice, twiss_0, nsuperperiod = 1):
    #TODO: add I4 for rectangular magnets I4 = Integrate(2 Dx(z)*k(z)*h(z), Z)

    n_points_element = 20

    tws_elem = twiss_0
    (I1, I2, I3,I4, I5) = (0., 0., 0., 0., 0.)
    h = 0.
    for elem in lattice.sequence:
        if elem.__class__ in (SBend, RBend, Bend) and elem.l != 0:
            Dx = []
            Hinvariant = []
            Z = []
            h = elem.angle/elem.l

            for z in linspace(0, elem.l,num = n_points_element, endpoint=True):
                tws_z = elem.transfer_map(z)*tws_elem
                Dx.append(tws_z.Dx)
                Z.append(z)
                Hx = (tws_z.gamma_x*tws_z.Dx*tws_z.Dx + 2.*tws_z.alpha_x*tws_z.Dxp*tws_z.Dx
                                        + tws_z.beta_x*tws_z.Dxp*tws_z.Dxp)
                Hinvariant.append(Hx)
            #H = array(h)
            H2 = h*h
            H3 = abs(h*h*h)
            I1 += h*simps(array(Dx), Z)
            I2 += H2*elem.l  #simps(H2, Z)*nsuperperiod
            I3 += H3*elem.l  #simps(H3, Z)*nsuperperiod
            I4 += h*(2*elem.k1 + H2)*simps(array(Dx), Z)
            I5 += H3*simps(array(Hinvariant), Z)
        tws_elem = elem.transfer_map*tws_elem
    #if abs(tws_elem.beta_x - twiss_0.beta_x)>1e-7 or abs(tws_elem.beta_y - twiss_0.beta_y)>1e-7:
    #    print( "WARNING! Results may be wrong! radiation_integral() -> beta functions are not matching. ")
        #return None
    return (I1*nsuperperiod,I2*nsuperperiod,I3*nsuperperiod, I4*nsuperperiod, I5*nsuperperiod)

class EbeamParams:
    def __init__(self, lattice, beam,  coupling = 0.01, nsuperperiod = 1, tws0 = None):
        if beam.E == 0:
            exit("beam.E must be non zero!")
        self.E = beam.E
        if tws0 == None:
            tws = twiss(lattice, Twiss(beam))
            self.tws0 = tws[0]
        else:
            #tws0.E = lattice.energy
            self.tws0 = tws0
            tws = twiss(lattice, tws0)

        self.lat = lattice
        (I1,I2,I3, I4, I5) = radiation_integrals(lattice, self.tws0 , nsuperperiod)
        self.I1 = I1
        self.I2 = I2
        self.I3 = I3
        self.I4 = I4
        self.I5 = I5
        #print "I2 = ", I2
        #print "I3 = ", I3
        #print "I4 = ", I4
        #print "I5 = ", I5
        self.Je = 2 + I4/I2
        self.Jx = 1 - I4/I2
        self.Jy = 1
        self.gamma = self.E/m_e_GeV
        self.sigma_e = self.gamma*sqrt(Cq * self.I3/(self.Je*I2))
        self.emittance = Cq*self.gamma*self.gamma * self.I5/(self.Jx* self.I2)
        self.U0 = Cgamma*(beam.E*1000)**4*self.I2/(2*pi)
        #print "*********  ", twiss_0.Energy
        self.Tperiod = nsuperperiod*lattice.totalLen/speed_of_light
        self.Length = nsuperperiod*lattice.totalLen
        self.tau0 = 2*self.E*1000*self.Tperiod/self.U0
        self.tau_e = self.tau0/self.Je
        self.tau_x = self.tau0/self.Jx
        self.tau_y = self.tau0/self.Jy
        self.alpha = self.I1/(speed_of_light*self.Tperiod)
        self.coupl = coupling
        self.emitt_x = self.emittance/(1 + self.coupl)
        self.emitt_y = self.emittance*self.coupl/(1 + self.coupl)
        self.sigma_x = sqrt((self.sigma_e*self.tws0.Dx)**2 + self.emitt_x*self.tws0.beta_x)
        self.sigma_y = sqrt((self.sigma_e*self.tws0.Dy)**2 + self.emitt_y*self.tws0.beta_y)
        self.sigma_xp = sqrt((self.sigma_e*self.tws0.Dxp)**2 + self.emitt_x*self.tws0.gamma_x)
        self.sigma_yp = sqrt((self.sigma_e*self.tws0.Dyp)**2 + self.emitt_y*self.tws0.gamma_y)


    def integrals_id(self):
        L = 0.
        self.I2_IDs = 0.
        self.I3_IDs = 0.
        self.I4_IDs = 0.
        self.I5_IDs = 0.
        for elem in self.lat.sequence:
            if elem.__class__ == Undulator:
                B = K2field(elem.Kx, lu = elem.lperiod)
                h0 = B*speed_of_light/self.E*1e-9
                #print h0, B
                tws = trace_z(self.lat, self.tws0, [L, L + elem.l/2.])
                i2 = I2_ID(elem.l,h0)
                i3 = I3_ID(elem.l,h0)
                i4 = I4_ID(elem.l,h0,elem.lperiod)
                i5 = I5_ID(elem.l,h0,elem.lperiod,tws[1].beta_x,tws[0].Dx, tws[0].Dxp)
                self.I2_IDs += i2
                self.I3_IDs += i3
                self.I4_IDs += i4
                self.I5_IDs += i5
                #print elem.type, elem.id, "B0 =  ", B, " T"
                #print elem.type, elem.id, "rho = ", 1./h0, " m"
                #print elem.type, elem.id, "L =   ", elem.l, " m"
                #print elem.type, elem.id, "beta_x cntr: ", tws[1].beta_x
                #print elem.type, elem.id, "Dx0 / Dxp0:  ", tws[0].Dx, "/", tws[0].Dxp
                #print elem.type, elem.id, "I2_ID = ", i2
                #print elem.type, elem.id, "I3_ID = ", i3
                #print elem.type, elem.id, "I4_ID = ", i4
                #print elem.type, elem.id, "I5_ID = ", i5
            L += elem.l
        self.emit_ID = self.emittance * (1.+self.I5_IDs/self.I5)/(1+(self.I2_IDs  - self.I4_IDs)/(self.I2 - self.I4))
        self.sigma_e_ID = self.sigma_e * sqrt((1.+ self.I3_IDs / self.I3)/(1 + (2*self.I2_IDs + self.I4_IDs)/(2.*self.I2 + self.I4) ) )
        self.U0_ID = Cgamma*(self.E*1000)**4.*self.I2_IDs/(2.*pi)
        print("emittance with IDs = ", self.emit_ID*1e9, " nm*rad")
        print("sigma_e with IDs =   ", self.sigma_e_ID)
        print("U0 from IDs =        ", self.U0_ID,  "  MeV")

    def __str__(self):
        val = ""
        val += ( "I1 =        " + str(self.I1) )
        val += ( "I2 =        " + str(self.I2) )
        val += ( "\nI3 =        " + str(self.I3) )
        val += ( "\nI4 =        " + str(self.I4) )
        val += ( "\nI5 =        " + str(self.I5) )
        val += ( "\nJe =        " + str(self.Je) )
        val += ( "\nJx =        " + str(self.Jx) )
        val += ( "\nJy =        " + str(self.Jy) )
        val += ( "\nenergy =    " + str(self.E) +"GeV")
        val += ( "\ngamma =     " + str(self.gamma) )
        val += ( "\nsigma_e =   " + str(self.sigma_e) )
        val += ( "\nemittance = " + str(self.emittance*1e9) +" nm*rad")
        val += ( "\nLength =    " + str(self.Length) + " m")
        val += ( "\nU0 =        " + str(self.U0) + "  MeV")
        val += ( "\nTperiod =   " + str(self.Tperiod*1e9) + " nsec")
        val += ( "\nalpha =     " + str(self.alpha) )
        val += ( "\ntau0 =      " + str(self.tau0*1e3) + " msec")
        val += ( "\ntau_e =     " + str(self.tau_e*1e3) + " msec")
        val += ( "\ntau_x =     " + str(self.tau_x*1e3) + " msec")
        val += ( "\ntau_y =     " + str(self.tau_y*1e3) + " msec")
        val += ( "\nbeta_x =    " + str(self.tws0.beta_x) + " m")
        val += ( "\nbeta_y =    " + str(self.tws0.beta_y) +" m")
        val += ( "\nalpha_x =   " + str(self.tws0.alpha_x))
        val += ( "\nalpha_y =   " + str(self.tws0.alpha_y))
        val += ( "\nDx =        " + str(self.tws0.Dx) + " m")
        val += ( "\nDy =        " + str(self.tws0.Dy) + " m")
        val += ( "\nsigma_x =   " + str(self.sigma_x*1e6) + " um")
        val += ( "\nsigma_y =   " + str(self.sigma_y*1e6) + " um")
        val += ( "\nsigma_x' =  " + str(self.sigma_xp*1e6) + " urad")
        val += ( "\nsigma_y' =  " + str(self.sigma_yp*1e6) + " urad\n")
        return val

from scipy.integrate import simps


'''
Damping time and diffusion coefficients
'''
def damp_rad(tws, lat, beam):
    eb = EbeamParams(lat, beam, nsuperperiod=1)

    di = type('dampInfo', (), {})
    #jx = 1.5
    jx = 2.93
    jy = 1.0
    #jz = 1.5
    jz = 0.06

    T0 = beam.T0 # sec, revolution time

    Cg = 8.846e-5 # m / Gev^3
    Cu = 55. / 24. / sqrt(3.)
    Cq = 3.823e-13
    gam = 1000. * beam.E / (0.511)

    hbar_c = 6.582 * 2.99e-8 # ev m

    r = 23 * 36 / pi; # machine radius

    s_irho_2 = [e.l / (e.l / e.angle)**2 for e in lat.sequence if e.__class__ in (SBend, RBend, Bend) and abs(e.angle) > 1.e-10]
    s_irho_3 = [e.l / (e.l / e.angle)**3 for e in lat.sequence if e.__class__ in (SBend, RBend, Bend) and abs(e.angle) > 1.e-10]
    l = tws[-1].s

    irho2_av = np.sum(s_irho_2) / l
    irho3_av = np.sum(s_irho_3) / l

    print(irho2_av)
    di.U0 = Cg * beam.E**4 * r * irho2_av # in GeV
    di.tau_z = 1. / (jz * di.U0 / (2*beam.E * T0) )
    di.tau_x = 1. / (jx * di.U0 / (2*beam.E * T0) )
    di.tau_y = 1. / (jy * di.U0 / (2*beam.E * T0) )

    di.Gx = 4. * (eb.I5 / eb.I2) * Cq * gam**2 / (di.tau_x * jx)
    #di.Gx = 4. * (eb.I5 / irho2_av*l) * Cq * gam**2 / (di.tau_x * jx)
    di.Ge = 3./2. * Cu * hbar_c * gam**3 * 1.e-9 * di.U0 / T0 * irho3_av /  irho2_av # in GeV^2

    di.ex = di.Gx * di.tau_x / 4.
    di.sige = sqrt(Cq * gam**2 * irho3_av  / jz / irho2_av)

    return di




from scipy.optimize import newton_krylov, broyden1, anderson, minimize
from scipy import optimize
'''
equilibrium emittance including IBS
'''
def emit(tws, lat, beam):

    di=damp_rad(tws, lat, beam)
    # objective
    def residual(x):
        #global tws, lat, beam, di
        beam2 = Beam(beam)
        beam2.kappa_h = beam.kappa_h
        beam2.kappa = beam.kappa
        beam2.tlen = beam.tlen
        beam2.E = beam.E
        beam2.N = beam.N
        beam2.emit_x = x[0]
        beam2.emit_y = x[1]
        beam2.sigma_E = x[2]
        beam2.alpha = beam.alpha
        beam2.ws = beam.ws

        beam2.tlen = beam2.alpha * 2.99e8 / beam2.ws * beam2.sigma_E

        r=ibs(tws, beam2, kappa_h=beam2.kappa_h)

        ex2 = di.Gx / (1./di.tau_x - r.Thm) / 4.
        ey2 = beam.kappa* di.Gx / (1./di.tau_y - r.Tvm) / 4.
        ez2 = di.Ge / (1./di.tau_z - r.Tem) / 4.

        if ex2 < 0 or ey2 < 0 or ez2 < 0 :
            return 1.e9  #pen_max

        fx = beam2.emit_x -  ex2
        fy = beam2.emit_y -  ey2
        ezm = (beam2.sigma_E * beam2.E)**2
        fz =  ezm - ez2
        #return np.array([fx**2,fy**2, fz**2])
        return (fx* 1.e12)**2 + (fy*1.e12)**2 + (fz*1.e6)**2


    guess = np.array([beam.emit_x, beam.emit_y, beam.sigma_E])
    #sol = newton_krylov(residual, guess, method='lgmres', verbose=1, x_tol=1.e-2)
    #sol = anderson(residual, guess, verbose=1, x_tol=1.e-2)
    #sol = optimize.fmin(residual, guess, xtol=1.e-3,maxiter=10000, maxfun=10000)
    sol = minimize(residual, guess, method='nelder-mead',options={'xtol': 1e-8, 'disp': False})

    print("solution", sol)
    print(sol["final_simplex"][0][0])

    beam_mod  = Beam()
    beam_mod.emit_x = sol["final_simplex"][0][0][0]
    beam_mod.emit_y = sol["final_simplex"][0][0][1]
    beam_mod.sigma_E = sol["final_simplex"][0][0][2]
    beam_mod.tlen = beam.alpha *2.99e8 / beam.ws * beam_mod.sigma_E

    return beam_mod




'''
IBS based on CIMP model (Kubo, Mtingwa and Wolski 2005)
'''

def g(x):
    if x<0.25: return -4.336*log(x)
    if x<0.8:  return 1./(x + 0.11)**1.6 + 1.2
    return 2./x**(1.2) - 0.3


def ibs(tws, beam, kappa_h=0.1):

    sigp = beam.sigma_E
    ex = beam.emit_x  # m
    ey = beam.emit_y  # m
    gam = beam.E * 1000./ 0.511
    sigs = beam.tlen # m
    N = beam.N
    L = tws[-1].s


    # if negative emittance passed, assume "very large" values
    if ex < 1.e-20: ex = 1.e-1
    if ey < 1.e-20: ey = 1.e-1
    if sigp < 1.e-20: sigp = 1.e-1


    print("start values ibs {} {} {}".format(ex,ey,sigp))

    b0 = 10.0 # average beta function -- for Coulomb log

    s = np.array([t.s for t in tws])
    betx = np.array([t.beta_x for t in tws])
    bety = np.array([t.beta_y for t in tws])
    H = np.array([(1. + t.alpha_x**2)/ t.beta_x * t.Dx**2 + t.beta_x*t.Dxp**2 + 2*t.alpha_x*t.Dx*t.Dxp  for t in tws])

    lattice_fudge = 1.0

    y0 = (1./sigp)**2 + lattice_fudge * np.abs(H)/ex

    y2 = 1. /sqrt(y0)
    sigh = y2
    a = sigh / gam * sqrt(betx/ex)
    b = sigh / gam * sqrt(bety/ey)
    g1 = [g(i) for i in b/a]
    g2 = [g(i) for i in a/b]


    kappa_c = 0.0
    fudge = 1.0

    #longitudinal
    Fe = sigh**2/sigp**2 * (g1/a + g2/b)
    #horizontal
    Fx = -a*g1 + fudge * H * sigh**2 / ex * (g1/a + g2/b)
    Fy = -b*g2 + fudge * kappa_h * H * sigh**2 / ey * (g1/a + g2/b)

    r0 = 2.8179403e-15
    r02c = 23.805e-22


    A = (r02c * N ) / (64.0 * pi**2 * gam**4 * ex*ex*sigp*sigs)
    LG = log( (gam**2 * ex * sqrt(b0 * ex)) / (r0 * b0) )

    Tem = 2. * pi**(3./2.)* LG * A * simps(Fe,s) / L
    Thm = 2. * pi**(3./2.)* LG * A * simps(Fx,s) / L
    Tvm = 2. * pi**(3./2.)* LG * A * simps(Fy,s) / L


    print('IBS rise times (x, y, E) [msec]:{} {} {}'.format(1000./Thm,1000./Tvm, 1000./Tem))
    print('IBS rise rates [1/sec]:{} {}'.format(Thm, Tem))

    ibsInfo = type('ibsInfo', (), {})
    ibsInfo.Fe = Fe
    ibsInfo.Fx = Fx
    ibsInfo.Fy = Fy
    ibsInfo.Thm = Thm
    ibsInfo.Tvm = Tvm
    ibsInfo.Tem = Tem

    ibsInfo.sigh = sigh
    ibsInfo.a = a
    ibsInfo.b = b
    ibsInfo.g1 = g1
    ibsInfo.g2 = g2
    ibsInfo.tws = tws
    ibsInfo.s = s
    ibsInfo.H = H


    return ibsInfo
