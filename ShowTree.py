#
# Display the branches in an EventNtuple TTree
#
import uproot
import awkward as ak
def Branches( file):
    with uproot.open(file) as rfile:
        print(rfile.keys())
#        rfile.arrays()
