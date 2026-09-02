"""
CPI Image Processing

This code processes individual png images output from Level1_HawkeyeCPI_individual_image_dump.py
and saves them in .csv format.

Modification History:
2024/01/12 - Joseph Finlon <joseph.a.finlon@nasa.gov>
    Initial code commit.
2024/06/12 - Christian Nairy <christian.nairy@und.edu>
    Added Compactness and Curl parameters.
    Also imported math to access sqrt.
"""
import glob
from multiprocessing import Pool
import tqdm
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
np.seterr(divide = 'ignore')
import xarray as xr
import pandas as pd
import cv2
from rembg import new_session, remove
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Ellipse
from skimage.filters import laplace
from scipy.ndimage import variance
from skimage.segmentation import clear_border
import sys
from math import sqrt
# , 'img_num': 'str'

# === SUBROUTINES ===
def proc_mp(filename):
    """
    Helper function that computes particle properties on a single roi image.

    Inputs:
    filename: str
        Full path to the roi image
    """
    schema={
        'time': 'datetime64[ns, UTC]', 'center_x': 'float64', 'center_y': 'float64',
        'dmax': 'float64', 'darea': 'float64', 'area': 'float64', 'perimeter': 'float64',
        'area_ratio': 'float64', 'aspect_ratio': 'float64', 'solidity': 'float64',
        'circularity': 'float64', 'fine_detail': 'float64', 'fractal_dimension': 'float64',
        'complexity': 'float64', 'intensity_range': 'float64',
        'ellipse_x': 'float64', 'ellipse_y': 'float64', 'ellipse_width': 'float64',
        'ellipse_height': 'float64', 'ellipse_major': 'float64', 'ellipse_angle': 'float64',
        'laplace_variance': 'float64', 'laplace_max': 'float64', 'pct_touching': 'float64',
        'compactness': 'float64', 'curl': 'float64', 'img_num': 'str'
    }
    df = pd.DataFrame(columns=schema.keys()).astype(schema)
    img_raw = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(
        cv2.bitwise_not(img_raw), 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU
    ) # https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    
    # morphological closing to close small gaps before contouring
    binary = cv2.bitwise_not(cv2.morphologyEx(
        cv2.bitwise_not(binary), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )) # https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
    
    # get particle properties
    _, props = get_img_props(filename, cv2.bitwise_not(binary), img_raw)
    if props:
        for prop in props:
            df.loc[len(df.time)] = list(prop)
            
    return df
    
# compute fractal dimension
def fractal_dimension(Z, threshold=50):
    '''
    Compute the 2D grayscale fractal dimension following the box count method.
    Follows https://stackoverflow.com/questions/44793221/python-fractal-box-count-fractal-dimension.
    Values should be < 2. Default threshold should yield stable frac dim based on testing from 1-100.
    Possible alternative: https://github.com/brian-xu/FractalDimension

    Inputs:
    Z: numpy.ndarray
        2-D grayscale (0-255) array output from cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    threshold: int
        Threshold used to transform the raw roi image into a binary array
    '''
    # only for 2d image
    assert(len(Z.shape) == 2)

    # from https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1
        )

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # transform Z into a binary array
    Z = (Z < threshold)

    # minimal dimension of image
    p = min(Z.shape)

    # greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # extract the exponent
    n = int(np.log(n)/np.log(2))

    # build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def get_img_props(image_filename, binary_image, raw_image):
    '''
    Derive image properties for storage in a pandas.DataFrame()

    Reference 1: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    Reference 2: https://opencv24-python-tutorials.readthedocs.io/en/latest/
        py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html

    Inputs:
    image_filename: str
        Full path to the roi image
    binary_image: numpy.ndarray
        2-D binary (0: shadowed | 1: background) array
    raw_image: numpy.ndarray
        2-D grayscale (0-255) array output from cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    '''
    # contour the perimeter, ignoring within the particle boundary
    binary_copy = binary_image.copy()
    contours, _ = cv2.findContours(
        binary_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    ) # for binary image derived from raw grayscale image (y,x)

    results = []
    boundaries = []
    for i, cnt in enumerate(contours): # loop through particles in scene (usually 1)
        # compute area for determining if object sufficient size
        area = cv2.contourArea(cnt) * (2.3 * 1.e-3)**2

        # get bounding rectangle to determine if large enough to get properties
        xR, yR, wR, hR = cv2.boundingRect(cnt)
        img_crop = raw_image[yR:yR+hR, xR:xR+wR] # focus on bounded region

        # only objects where Deq >= 30 um and roi image larger than 3x3 pixels
        if (area>=np.pi*0.015**2) and (img_crop.shape[0] >=75) and (
                img_crop.shape[1] >=75):

            # time information
            dt_str = (
                f'{image_filename.split("_")[-3][:8]}T{image_filename.split("_")[-3][14:20]}.'
                f'{image_filename.split("_")[-3][20:23]}'
            )
            # print(dt_str)
            dt = pd.Timestamp(dt_str, tz='UTC')

            # image_number
            img_num = f'{image_filename.split("_")[-2][:]}'
            #print(img_num)

            # perimeter, solidarity and circularity
            perimeter = cv2.arcLength(cnt, True) * 2.3 * 1.e-3
            hull = cv2.convexHull(cnt)
            solidity = float(area) / (cv2.contourArea(hull) * (2.3 * 1.e-3)**2)
            circularity = 4 * np.pi * area / perimeter**2
    

            #Compactness - http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
            # C = perimeter**2/(4*pi*area)
            compactness = perimeter**2 / (4 * np.pi * area)

            # area-equivalent diam, area ratio and find-detail ratio from minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            deq = 2. * np.sqrt(area / np.pi)
            fr = perimeter * (2 * radius * 2.3 * 1.e-3) / area
            ar = area / (np.pi * (radius * 2.3 * 1.e-3)**2)
            
            #Curl - http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
            # curl = length (or dmax) / fiber length
            # fiber length = (perimeter - ((perimeter**2) - 16 * area)**(1/2)) / 4
            
            #fiber_length = (perimeter - sqrt((perimeter**2) - 16 * area)) / 4
            #curl = (2 * radius * 2.3 * 1.e-3) / float(fiber_length)
            #test = (perimeter**2) - 16 * area
            perimeter_squared = perimeter ** 2
            inner_value = perimeter_squared - 16 * area
    
            if inner_value < 0:  # Check if the value inside the square root is negative
                curl = 99999.999
            else:
                fiber_length = (perimeter - sqrt(inner_value)) / 4 
                curl = (2 * radius * 2.3 * 1.e-3) / float(fiber_length)

            # aspect ratio from ellipse fit
            # fit uses least square optimization that assumes the points to lie on an ellipse
            # https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ellipse-pami.pdf
            if len(hull) >= 5: # more accurate ellipse fit with hull (needs 5 pts though)
                ellipse = cv2.fitEllipse(hull)
            else:
                ellipse = cv2.fitEllipse(cnt)
            ell_x, ell_y = ellipse[0]
            ell_width, ell_height = ellipse[1]
            ell_major = np.max(ellipse[1]) * 2.3 * 1.e-3
            ell_angle = ellipse[2]
            asr = np.min(ellipse[1]) / np.max(ellipse[1])
    
            # fractal dimension
            # https://stackoverflow.com/questions/44793221/python-fractal-box-count-fractal-dimension
            fd = fractal_dimension(raw_image, threshold=50)

            # particle complexity (closer to 1: circular, rimed; higher: aggregate, less rimed)
            # Fitch and Garrett 2022 (https://doi.org/10.1029/2021JD035980)
            # Garrett and Yuter 2014 (https://doi.org/10.1002/2014GL061016)
            window_max = np.max(
                sliding_window_view(img_crop, window_shape=(3, 3)), axis=(2, 3)
            ) # max intensity in 3x3 window surrounding each pixel
            window_min = np.min(
                sliding_window_view(img_crop, window_shape=(3, 3)), axis=(2, 3)
            ) # min intensity in 3x3 window surrounding each pixel
            window_rg = (window_max - window_min) / 255. # intensity range, normalized from 0-1
            sigma = np.mean(window_rg, axis=(0, 1)) # mean interpixel variability
            chi = perimeter * (1. + sigma) / np.pi / deq # FG22 Eqn 1

            # quantify blurriness
            # https://medium.com/snapaddy-tech-blog/mobile-image-blur-detection-with-machine-learning-c0b703eab7de
            edge_laplace = laplace(raw_image, ksize=3)
            lap_var = variance(edge_laplace)
            lap_max = np.amax(edge_laplace)

            # fraction of image perimeter on edge of frame
            xs = np.array([v[0][0] for v in cnt])
            ys = np.array([v[0][1] for v in cnt])
            frac_touching = (
                sum(ys == 0) + sum(ys + 1 == binary_image.shape[0])
                + sum(xs == 0) + sum(xs + 1 == binary_image.shape[1])
            ) / (perimeter / 0.0023) # num pixels touching edge / perimeter
            
            results.append(
                (
                    dt, x, y, 2 * radius * 2.3 * 1.e-3, deq, area, perimeter, ar, asr,
                    solidity, circularity, fr, fd, chi, sigma, ell_x, ell_y,
                    ell_width, ell_height, ell_major, ell_angle,
                    lap_var, lap_max, 100. * frac_touching, compactness, curl, img_num
                )
            )
            boundaries.append(cnt)
    if results:
        return boundaries, results
    else:
        return None, None

if __name__ == '__main__':
    """
    Main code to process the CPI data for a flight.
    Takes three arguments to grab the data files and save the output.
    Parallel processing is used here, but this section can be reworked easily
    to process one image at a time.

    Inputs:
    date: Flight start date in %Y-%m-%d format
    indir: Path to the parent directory storing the CPI images (excludes trailing "/")
    outdir: Path to save the particle properties .csv file (excludes trailing "/")
    nproc: Number of processors to use in parallel computing

    Syntax:
    python cpi-img_process.py 2022-01-19 /path/to/source_workspace/plots/IMPACTS/CPI_Images_Individual 
        /path/to/source_workspace/data/IMPACTS/CPI_Particle_Properties 40
    """
    # populate user inputs
    date = sys.argv[1]
    datestr = date.replace('-', '')
    indir = sys.argv[2]
    outdir = sys.argv[3]
    nproc = int(sys.argv[4])
    
    files = sorted(glob.glob(f'{indir}/{datestr}/**/*.png', recursive=True))
    # print(files)
   
    p = Pool(processes=nproc)

    # this block does essentailly all of the processing
    data = pd.concat(p.map(proc_mp, tqdm.tqdm(files, total=len(files))))
    p.close()
    p.join()

    # assign unique times to multiple particles in a roi png
    td = pd.to_timedelta(data.groupby('time').cumcount(), unit='us')
    data.time = data.time + td

    # order the particles by time (parallel processing thows things everywhere)
    data = data.set_index('time', drop=True).sort_index()
    outfile = (
        f'{outdir}/cpi.particle_properties_Nairy.{datestr}.csv'
    )
    data.to_csv(outfile)
