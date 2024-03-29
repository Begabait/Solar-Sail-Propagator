# This is the primary Python file which is used to set the parameters, run the scripts and set the visualizations

from RK_Driver import Test_Clib_Interface, Set_Pend_coeff, DP45_Integrator, RK4_Integrator
from numpy import pi
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from Visualizations import plot_2D_time, parse_results_doublep, plot_2D_phase, plot_energy
from scipy import interpolate

# Get the integrator parameters and constants based on physical pendulum dimensions
# Inputs:
# - l1, l2 are the lengths of the upper rod (l1) and lower rod (l2) in [m]
# - m1, m2 are the masses of the upper rod (m1) and lower rod (m2) in [kg]
# - g is the gravitational acceleration g = 9.8 [m/s^2] for Earth surface gravity
# Outputs:
# - coeff_vals[ 5 ] double array contains the coefficients a_th to b_phi in sequence 
# NOTE: Order of the output is [ a_th , a_phi , a_mix , b_th , b_phi ] as per RK_Library.h
def get_int_params( l1 , l2 , m1 , m2 , g_acc ):

    # Compute the pendulum parameters and return them as a 5D array - based on eq.(9-13) from the .pdf
    pend_param = [ m1*l1*l1/6.0 + 0.5*m2*l2*l2 ,
                   m2*l2*l2/6.0 ,
                   0.5*m2*l1*l2 ,
                   l1*g_acc*( m1/2.0 + m2 ) ,
                   0.5*l2*g_acc*m2 ]

    return pend_param

# Animate function for the physical visualization of the pendulum 
# Inputs:
# - int i_anim: index of the animation (indexing the arrays)
# Outputs:
# - Outputs the anim lines 
def animate_phys( i ):

    # If we reach the end of the loop - start over
    if i == 0:
        x1.clear( )
        y1.clear( )
        x2.clear( )
        y2.clear( )
    
    # Append the latest values for both points
    x1.append( x_mid[ i ] )
    y1.append( y_mid[ i ] + ltot )
    x2.append( x_end[ i ] )
    y2.append( y_end[ i ] + ltot )

    # Create the lists for plots, adding the lines connecting the point [ 0 , l1 + l2 ] with the midpoint and the mid-to-end points line
    xlist = [ x1 , x2 , [ 0 , x_mid[ i ] ] , [ x_mid[ i ] , x_end[ i ] ] ]
    ylist = [ y1 , y2 , [ ltot , y_mid[ i ] + ltot ] , [ y_mid[ i ] + ltot , y_end[ i ] + ltot ] ]

    #for index in range(0,1):
    for lnum,line in enumerate( lines ):
        line.set_data( xlist[ lnum ] , ylist[ lnum ] ) # set data for each line separately. 

    return lines

# Initialize the physical visualization of the pendulum plot
def init_phys( ):
    for line in lines:
        line.set_data( [ ] , [ ] )
    return lines

if __name__ == "__main__":

    #########################################################
    # Pendulum and Physical Parameters to modify
    #########################################################
    l1 = 0.3 # in [m]
    l2 = 0.3 # in [m]
    m1 = 0.1 # in [kg]
    m2 = 0.1 # in [kg]
    g_acc = 9.8 # gravitational acceleration in [m/s^2]
    #########################################################
    # Simulation Initial and Accuracy Parameters to modify
    #########################################################
    err_tol = 1e-12 # Error Tolerance for the integrator
    range_int = [ 0.0 , 6.0*pi ] # Time range of integration in [sec]
    range_int = [0.0, 6000]
    state_init = [ 1.0 , pi , 0.0 , 0.0 ] # Initial state [ theta_0 , phi_0 , om_th_0 , om_phi_0 ]
    state_init = [6771000, 0, 0, 0, 5.5e3, 5.5e3]
    #########################################################
    # Output File details
    #########################################################
    out_file = b"Test_Results.csv" # Filename for the output - must be binary
    header = b"T [time], Theta [rad], Phi [rad], Om_Theta [rad/s], Om_Phi [rad/s]" # Header for the output file
    header = b"T [time], x [m], y [m], z [m], vx [m/s], vy [m/s], vz [m/s]"
    #########################################################

    ltot = l1 + l2
    pend_par = get_int_params( l1 , l2 , m1 , m2 , g_acc )

    # Test library and initialize coefficients
    Test_Clib_Interface( pi )

    Set_Pend_coeff( pend_par )

    DP45_Integrator( err_tol , state_init , range_int , out_file , header )

    # time, theta, phi, om_theta, om_phi = parse_results_doublep( out_file )
    time, x, y, z, vx, vy, vz = parse_results_doublep( out_file )

    # Compute the energy of the Pendulum (Kinetic, Potential and Total)
    e_kin = [ 0 ]*len( time )
    e_pot = [ 0 ]*len( time )
    # for j in range( 0 , len( time ) ):
    #     e_kin[ j ] = pend_par[ 0 ]*om_theta[ j ]*om_theta[ j ] + pend_par[ 1 ]*om_phi[ j ]*om_phi[ j ] + pend_par[ 2 ]*np.cos( phi[ j ] - theta[ j ] )*om_theta[ j ]*om_phi[ j ]
    #     e_pot[ j ] = - pend_par[ 3 ]*np.cos( theta[ j ] ) - pend_par[ 4 ]*np.cos( phi[ j ] )
    
    #########################################################
    # Static plot of some of the results
    #########################################################
    # - 2D static plot of the angles as a function of time -> change last param to save as .pdf
    plot_2D_time( time , x , y , vx , vy , "0" )
    # - 2D static plot of the 2D phase space slices -> change last param to save as .pdf
    plot_2D_phase( x , y , vx , vy , "0" )
    # - 2D static plot of the energy as a function of time -> change last param to save as .pdf
    plot_energy( time , e_kin , e_pot , "0" )
    #########################################################

    #########################################################
    # Animation of the physical pendulum results
    #########################################################

    # Convert to physical positions of the end points of the first and second pendulum
    # x_mid_0 = l1*np.sin( theta )
    # y_mid_0 = - l1*np.cos( theta )
    # x_end_0 = x_mid_0 + l2*np.sin( phi )
    # y_end_0 = y_mid_0 - l2*np.cos( phi )
    x_mid_0 = l1*np.sin( x )
    y_mid_0 = - l1*np.cos( x )
    x_end_0 = x_mid_0 + l2*np.sin( y )
    y_end_0 = y_mid_0 - l2*np.cos( y )

    # Create a uniform time array with the range of time and points corresponding to roughly 100 per second of physical time
    time_int = np.linspace( min( time ) , max( time ) , 100*int( max( time ) ) )
    # Create the tck handles for the different functions
    tck_xmid = interpolate.splrep( time , x_mid_0 , s = 0 )
    tck_ymid = interpolate.splrep( time , y_mid_0 , s = 0 )
    tck_xend = interpolate.splrep( time , x_end_0 , s = 0 )
    tck_yend = interpolate.splrep( time , y_end_0 , s = 0 )

    # Interpolate with the points based on time_int 
    x_mid = interpolate.splev( time_int , tck_xmid , der = 0 )
    y_mid = interpolate.splev( time_int , tck_ymid , der = 0 )
    x_end = interpolate.splev( time_int , tck_xend , der = 0 )
    y_end = interpolate.splev( time_int , tck_yend , der = 0 )
    
    fig = plt.figure( )
    ax1 = plt.axes( xlim = ( min( [ min( x_end ) , min( x_mid ) ] ) - 0.1 , max( [ max( x_end ) , max( x_mid ) ] ) + 0.1 ) , 
                    ylim = ( - 0.1 , max( [ max( y_end ) , l1 + l2 ] ) + 0.1 ) )
    line, = ax1.plot( [ ] , [ ] , lw = 1 )
    plt.xlabel( "X [m]" )
    plt.ylabel( "Y [m]" )

    plotlays, plotcols, lwvals = [ 4 ], [ "green" , "red" , "black" , "blue" ], [ 1 , 1 , 2 , 2 ]
    lines = []

    for index in range( 4 ):
        lobj = ax1.plot( [ ] , [ ] , lw = lwvals[ index ] , color = plotcols[ index ] )[ 0 ]
        lines.append( lobj )

    x1, y1 = [], []
    x2, y2 = [], []

    frame_num = len( time_int )

    # Call the animator, blit=True means only re-draw the parts that have changed
    anim = animation.FuncAnimation( fig , animate_phys , init_func = init_phys , frames = frame_num , interval = 10 , blit = True )

    # Currently gif comes out huge, should reduce its size
    #anim.save( "Test_Gif.gif" , writer = animation.PillowWriter( fps = 60 ) )

    plt.show()
    #########################################################
