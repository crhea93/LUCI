"""
Suite of tests for Luci to test how she reads a cube, extracts a spectrum, fits and so on.
This only tests that all the functions are working -- not necessarily giving the correct results.
"""
from LuciBase import Luci


class Test:
    def __init__(self):
        # Set Parameters
        Luci_path = '/home/carterrhea/Documents/LUCI/'
        cube_dir = '/mnt/carterrhea/carterrhea/NGC6946/NGC6946_SN3'  # Path to data cube
        cube_name = 'NGC6946_SN3'  # don't add .hdf5 extension
        object_name = 'NGC6946'
        redshift = 0.000133
        resolution = 5000
        # Read in cube
        self.cube = Luci(Luci_path, cube_dir + '/' + cube_name, cube_dir, object_name, redshift, resolution)



def test_read_cube():
    """
    Test to call reading in a cube. By initializing the Test class, a cube will be read in.
    """
    Test_ = Test()
    assert len(Test_.cube.cube_final.shape) == 3


def test_create_deep_frame():
    """
    Test to call reading in a cube. By initializing the Test class, a cube will be read in.
    """
    cube = Test().cube
    cube.create_deep_image()
    assert len(cube.deep_image.shape) == 2


def test_bin_cube():
    """
    Test to call bin cube. We do 2x2 binning on a region of the cube (10<x<20, 10<y<20)
    """
    cube = Test().cube
    cube.bin_cube(binning=2, x_min=10, x_max=20, y_min=10, y_max=20, cube_final=cube.cube_final, header=cube.header)
    assert cube.cube_binned.shape[0] == 5
    assert cube.cube_binned.shape[1] == 5

def test_fit_cube_sincgauss():
    """
    Test to make sure that fit_cube() works
    """
    cube = Test().cube
    lines = ['Halpha']
    fit_function = 'sincgauss'
    vel_rel = [1]
    sigma_rel = [1]
    x_min = 100
    x_max = 101
    y_min = 100
    y_max = 101
    cube.fit_cube(lines, fit_function, vel_rel, sigma_rel,
             x_min, x_max, y_min, y_max)


def test_fit_cube_gauss():
    """
    Test to make sure that fit_cube() works
    """
    cube = Test().cube
    lines = ['Halpha']
    fit_function = 'gaussian'
    vel_rel = [1]
    sigma_rel = [1]
    x_min = 100
    x_max = 101
    y_min = 100
    y_max = 101
    cube.fit_cube(lines, fit_function, vel_rel, sigma_rel,
             x_min, x_max, y_min, y_max)


def test_fit_cube_sinc():
    """
    Test to make sure that fit_cube() works
    """
    cube = Test().cube
    lines = ['Halpha']
    fit_function = 'sinc'
    vel_rel = [1]
    sigma_rel = [1]
    x_min = 100
    x_max = 101
    y_min = 100
    y_max = 101
    cube.fit_cube(lines, fit_function, vel_rel, sigma_rel,
             x_min, x_max, y_min, y_max)
