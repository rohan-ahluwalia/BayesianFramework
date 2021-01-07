#//=============================================================
#// @author rohanahluwalia
#//=============================================================

"""Here is an example of the code that will run all the functions

This code will provide an entire bayesian network for the loaded system file"""

import bnomics
import sys

# First load the datafile
filename=sys.argv[1]
dt=bnomics.dutils.loader(filename)

##############################
# Now discretize the variables with
dt.discretize_all())

#######################
# Initialize the search
srch=bnomics.search(dt)

####################
# Perform the search
srch.gsrestarts()

################################################################################
# Save the reconstructed BN structure in dotfile.dot and generate a rendering of
# the result in outpdf.pdf (if Graphviz is properly installed).
srch.dot()


