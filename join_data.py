import os
import sys
import logging
logger = logging.getLogger(__name__)

import dedalus.public
from dedalus.tools  import post

data_dir = sys.argv[1]
base_path = os.path.abspath(data_dir)+'/'

logger.info("joining data from Dedalus run {:s}".format(data_dir))

data_types = ['checkpoint', 'scalar', 'profiles', 'slices', 'coeffs']

for data_type in data_types:
    logger.info("merging {}".format(data_type))
    try:
        post.merge_analysis(base_path+data_type)
    except:
        logger.info("missing {}".format(data_type))
        
logger.info("done join operation for {:s}".format(data_dir))
