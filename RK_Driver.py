# This driver is used to simplify the calls of the C shared library
# Its functions act as an interface for ctypes conversion:
# Simply import it into any other Python code (from the respective directory) and run the functions as described!

from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
import csv

# Import the shared RK library
lib_RK = CDLL( "./srp.dll" )

# IMPORTANT: Must be done before running any integration !!!
# Initialize all the RK coefficients for the Dormand-Prince method !!!
lib_RK.Set_RK_Coeff( )

# Printout the Runge-Kutta constants for integration to check that they are set appropriately
def Check_RK_Coeff( ):

    lib_RK.Check_RK_Coeff( )

# Test interface to the C library from Py 
# Enter double x value to be allocated and check that it is true 
def Test_Clib_Interface( x_val ):

    lib_RK.Test_Interface( c_double( x_val ) )

# Set the pendulum coefficients for the RHS computation
# Inputs:
# - coeff_vals[ 5 ] double array contains the coefficients a_th to b_phi in sequence
# NOTE: This function must be called before starting an integration or all the constants will be defaulted to 1.0 
def Set_Pend_coeff( coeff_vals ):

    lib_RK.Set_Pend_coeff.restype = None
    lib_RK.Set_Pend_coeff.argtypes = [ ndpointer( c_double ) ]
    lib_RK.Set_Pend_coeff( np.array( coeff_vals ) )

# 4th order Runge-Kutta integrator for testing purposes
# Inputs:
# - Npoints: number of integration points (NOT INTERVALS)
# - state_init[ dim_state ]: initial state for the integrator
# - range_int[ 2 ]: initial and final values evolution parameter (initial and final time)
# - file_name: a char of the output filename where the results will be written - include .csv in this like "file.csv"
# - header: a char of the header to start the file with (no need for \n sign) */
# Outputs:
# - The results are written in a file as commas separated values (.csv)
# -- The format is [ Time , State[ 0 ] , State[ 1 ] , ... State[ Nstate - 1 ] ]
# -- Reflect this in the header format!
# NOTE: Currently this function is linked to a RHS of a single Harmonic Oscillator with angular frequency \omega = 1.0
def RK4_Integrator( npoints , state_init , range_int , file_name , header ):

    nstate = len( state_init )

    lib_RK.RK4_Integrator.restype = None
    lib_RK.RK4_Integrator.argtypes = [ c_int , c_int , ndpointer( c_double ) , ndpointer( c_double ) , c_char_p , c_char_p ]
    lib_RK.RK4_Integrator( nstate , npoints , np.array( state_init ) , np.array( range_int ) , file_name , header )

# 4-5th order adaptive Dormand-Prince integrator 
# Inputs:
# - err_tol: error tolerance per step -> the adaptive step is modified to maintain this
# - state_init[ dim_state ]: initial state for the integrator
# - range_int[ 2 ]: initial and final values evolution parameter (initial and final time)
# - file_name: a char of the output filename where the results will be written - include .csv in this like "file.csv"
# - header: a char of the header to start the file with (no need for \n sign) 
# Outputs:
# - The results are written in a file as commas separated values (.csv)
# -- The format is [ Time , State[ 0 ] , State[ 1 ] , ... State[ Nstate - 1 ] ]
# -- Reflect this in the header format!
def DP45_Integrator( err_tol , state_init , range_int , file_name , header ):

    nstate = len( state_init )

    lib_RK.DP45_Integrator.restype = None
    lib_RK.DP45_Integrator.argtypes = [ c_int , c_double , ndpointer( c_double ) , ndpointer( c_double ) , c_char_p , c_char_p ]
    lib_RK.DP45_Integrator( nstate , err_tol , np.array( state_init ) , np.array( range_int ) , file_name , header )


def parse_results_doublep(filename):
    fp = open(filename, "r")

    if fp.readable():
        data = csv.reader(fp)
        lst = []
        for line in data:
            lst.append(line)
        ndata = len(lst) - 1

        time = [0] * ndata
        x = [0] * ndata
        y = [0] * ndata
        z = [0] * ndata
        vx = [0] * ndata
        vy = [0] * ndata
        vz = [0] * ndata

        for i in range(0, ndata):
            # time[ i ] = float( lst[ i + 1 ][ 0 ] )
            # theta[ i ] = float( lst[ i + 1 ][ 1 ] )
            # phi[ i ] = float( lst[ i + 1 ][ 2 ] )
            # om_theta[ i ] = float( lst[ i + 1 ][ 3 ] )
            # om_phi[ i ] = float( lst[ i + 1 ][ 4 ] )
            time[i] = float(lst[i + 1][0])
            x[i] = float(lst[i + 1][1])
            y[i] = float(lst[i + 1][2])
            z[i] = float(lst[i + 1][3])
            vx[i] = float(lst[i + 1][4])
            vy[i] = float(lst[i + 1][5])
            vz[i] = float(lst[i + 1][6])
    else:
        print("Unreadable data, something's wrong with the file " + filename)

    return time, x, y, z, vx, vy, vz