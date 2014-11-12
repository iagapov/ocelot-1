'''
user interface for viewing/editing photon optics layouts
'''

from numpy import sin, cos, pi, sqrt, log, array, random, sign
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
#import matplotlib.animation as animation

from ocelot.optics.elements import *
from ocelot.optics.wavefront import *

def init_plots(views, geo):
    scene = Scene()
    scene.views = views
    scene.fig = plt.figure()
    
    nviews = len(views)
    
    scene.ax = ['']*nviews
    scene.profile_im = {}
    
    for iview in xrange(nviews):
        view_id = nviews*100 + 10 + iview + 1
        scene.ax[iview] = scene.fig.add_subplot(view_id, autoscale_on=True)

        if views[iview] == 'geometry:x' or views[iview] == 'geometry:y':
            #scene.line_wf, = scene.ax[iview].plot([], [], '.', lw=2)
            plot_geometry(scene.ax[iview], geo)
            scene.ax[iview].grid()
            projection_name = views[iview].split(':')[1]
            scene.ax[iview].text(0.15, 0.85, projection_name,
                                 horizontalalignment='left',
                                 verticalalignment='top',
                                 transform=scene.ax[iview].transAxes)
            #scene.time_text = scene.ax[iview].text(0.05, 0.9, '', transform=scene.ax[iview].transAxes)

        if views[iview].startswith('detectors'):
            if views[iview].startswith('detectors:'):
                id = views[iview].split(':')[1]
                for obj in geo():
                    if obj.__class__ == Detector:
                        print 'adding view for detector: ', obj.id
                        scene.ax[iview].set_title('detector:' + id)
                        scene.profile_im[id] = scene.ax[iview] 
                        #scene.profile_im[id] = scene.ax[iview].imshow(obj.matrix.transpose(), cmap='gist_heat',interpolation='none',extent=[0,1,0,1], vmin=0, vmax=10)

    return scene


def plot_geometry(ax, geo, scales = [1,1,1]):
    
    for o in geo():
        if o.__class__ == Mirror:
            
            #TODO; replace with generic rotation
            ang = - np.arctan2(o.no[1] , o.no[2]) - pi
            #print 'ang=', ang
            z1 = o.r[2] - o.size[1] * sin(ang)
            z2 = o.r[2]
            z3 = o.r[2] + o.size[1] * sin(ang)
            y1 = -o.size[1] + o.r[1] + o.size[1]*(1-cos(ang))
            y2 = o.r[1]
            y3 = o.size[1] + o.r[1] - o.size[1]*(1-cos(ang))
            li, = ax.plot([z1,z2,z3], [y1,y2,y3], 'b-', lw=3)

            y_bnd = np.linspace(y1,y3, 100)
            z_bnd = np.linspace(z1, z3, 100)

            for z,y in zip(z_bnd[5::10],y_bnd[5::10]):
                tick_size = o.size[2]
                ax.plot([z,z-tick_size*np.sign(o.no[2])], [y,y-(y_bnd[5] - y_bnd[0])], 'b-', lw=2)

        if o.__class__ == EllipticMirror:
            
            #TODO; replace with generic rotation
            ang = - np.arctan2(o.no[1] , o.no[2]) - pi
            #print 'ang=', ang
            z1 = o.r[2] - o.size[1] * sin(ang)
            z2 = o.r[2]
            z3 = o.r[2] + o.size[1] * sin(ang)
            y1 = -o.size[1] + o.r[1] + o.size[1]*(1-cos(ang))
            y2 = o.r[1]
            y3 = o.size[1] + o.r[1] - o.size[1]*(1-cos(ang))
            #li, = ax.plot([z1,z2,z3], [y1,y2,y3], color="#aa00ff", lw=3)

            phi_max = np.arcsin(o.size[1]/o.a[0])

            #y_bnd = np.linspace(y1,y3, 100)
            phi_bnd = np.linspace(-phi_max, phi_max, 100)
            z_bnd = np.zeros_like(phi_bnd)
            y_bnd = np.zeros_like(phi_bnd)

            for i in xrange( len(phi_bnd) ):
                z_bnd[i] = o.r[2] + o.a[0]*sin(phi_bnd[i]) 
                y_bnd[i] = o.r[1] + o.a[1] - o.a[1]*cos(phi_bnd[i])

            #for z,y in zip(z_bnd[5:-10:10],y_bnd[5:-10:10]):
            n_step = 2
            for i in np.arange(0,len(z_bnd) - n_step ,n_step):
                tick_size = o.size[2]
                #ax.plot([z,z-tick_size*np.sign(o.no[2])], [y,y-(y_bnd[5] - y_bnd[0])], 'b-', lw=2)
                ax.plot([z_bnd[i],z_bnd[i+n_step]], [y_bnd[i],y_bnd[i+n_step]], color="#aa00ff", lw=3)


        if o.__class__ == ParabolicMirror:
            
            y_bnd = np.linspace(-o.size[1], o.size[1], 100)
            z_bnd = o.r[2] - o.a[1] * y_bnd**2

            #print y_bnd, z_bnd
            li, = ax.plot(z_bnd, y_bnd, 'b-', lw=3)
            
            for z,y in zip(z_bnd[5::10],y_bnd[5::10]):
                ax.plot([z,z-1.0*np.sign(o.no[2])], [y,y-(y_bnd[5] - y_bnd[0])], 'b-', lw=2)

        
        if o.__class__ == Lense:
            
            y_bnd = np.linspace(-o.D/2,o.D/2,100)
            z_bnd1 = (o.r[2]-o.s1) + (o.s1 / (o.D/2)**2 ) * y_bnd**2
            z_bnd2 = (o.r[2]+o.s2) - (o.s2 / (o.D/2)**2 ) * y_bnd**2
            li, = ax.plot(z_bnd1, y_bnd, 'r-', lw=3)
            li, = ax.plot(z_bnd2, y_bnd, 'r-', lw=3)

        if o.__class__ == Aperture:
            
            li, = ax.plot([o.r[2],o.r[2]], [o.r[1] + o.d[1],o.r[1] + o.size[1]], color='#000000', lw=3)
            li, = ax.plot([o.r[2],o.r[2]], [o.r[1] -o.d[1],o.r[1] - o.size[1]], color='#000000', lw=3)

        if o.__class__ == Crystal:
            
            li, = ax.plot([o.r[2],o.r[2]], [o.r[1] - o.size[1], o.r[1] + o.size[1]], color='#999999', lw=3)
            
        if o.__class__ == Grating:
            #TODO; replace with generic rotation
            ang = - np.arctan2(o.no[1] , o.no[2]) - pi
            #print 'ang=', ang
            z1 = o.r[2] - o.size[1] * sin(ang)
            z2 = o.r[2]
            z3 = o.r[2] + o.size[1] * sin(ang)
            y1 = -o.size[1] + o.r[1] + o.size[1]*(1-cos(ang))
            y2 = o.r[1]
            y3 = o.size[1] + o.r[1] - o.size[1]*(1-cos(ang))
            li, = ax.plot([z1,z2,z3], [y1,y2,y3], color="#AA3377", lw=3)

            y_bnd = np.linspace(y1,y3, 100)
            z_bnd = np.linspace(z1, z3, 100)

            dy = max(abs(y3-y1), abs(z3-z1)) / 20
            dz = dy
            
            for z,y in zip(z_bnd[5::10],y_bnd[5::10]):
                
                ax.plot([z-dz,z,z+dz], [y,y+dy, y], color="#AA3377", lw=2)


            
        zmax = np.max( map(lambda x: x.r[2] + x.size[2],geo))
        zmin = np.min( map(lambda x: x.r[2] - x.size[2],geo))

        ymax = np.max( map(lambda x: x.r[1] + x.size[1],geo))
        ymin = np.min( map(lambda x: x.r[1] - x.size[1],geo))

        z_margin = (zmax - zmin)*0.1
        y_margin = (ymax - ymin)*0.1
        
        #print zmin, zmax, z_margin, ymin, ymax, y_margin 
                
        #ax.set_xlim(zmin-z_margin,zmax+z_margin)
        #ax.set_ylim(ymin-y_margin,ymax+y_margin)



def plot_rays(ax, rays, proj='x'):
    
    for r in rays:
        debug('plotting ray!', r.r0[0], r.k[0], r.s[0])
        for i in xrange(len(r.r0)):
            debug('-->', r.r0[i], r.k[i], r.s[i])
            if proj == 'x':
                ax.plot([r.r0[i][2], r.r0[i][2] + r.k[i][2]*r.s[i] ], [r.r0[i][0], r.r0[i][0] + r.k[i][0]*r.s[i] ], color='#006600', lw=1, alpha=0.4 )
            if proj == 'y':
                ax.plot([r.r0[i][2], r.r0[i][2] + r.k[i][2]*r.s[i] ], [r.r0[i][1], r.r0[i][1] + r.k[i][1]*r.s[i] ], color='#006600', lw=1, alpha=0.4 )
    