#pragma once

// do not modify cvoro_config.h, it is build from cvoro_config.h.in
/* #undef USE_DOUBLE */

#ifndef USE_DOUBLE
#  define real float
#  define real4 cl_float4
#else
#  define real double
#  define real4 cl_double4
#endif

#  define CVORO_STATUS_FILE "/home/hyan/Projects/GPU-Restricted-Power-Diagrams/Status.h"
#  define CVORO_CONVEX_CELL_FILE "/home/hyan/Projects/GPU-Restricted-Power-Diagrams/ConvexCell.cl"
#  define CVORO_KNEAREST_FILE "/home/hyan/Projects/GPU-Restricted-Power-Diagrams/knearests.cl"
#  define CVORO_OPTIONS_FILE "/home/hyan/Projects/GPU-Restricted-Power-Diagrams/cvoro_options.txt"

