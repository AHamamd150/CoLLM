import uproot
import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt
import sys

vector.register_awkward()

def main(input_filename):
    tree = uproot.open(input_filename)
    selected_branches = ["pt", "eta", "phi", "mass"]
    data = tree.arrays(selected_branches, library="ak")  # Added library="ak"
    
    # Apply multiplicity cut
    n_particles = ak.num(data["pt"])
    event_mask = (n_particles == 2)
    data = data[event_mask]  # Apply event mask
    
    pt = data["pt"]
    eta = data["eta"]
    phi = data["phi"]
    mass = data["mass"]
    
    # Calculate invariant mass
    p1 = vector.zip({"pt": pt[:, 0], "eta": eta[:, 0], "phi": phi[:, 0]})
    p2 = vector.zip({"pt": pt[:, 1], "eta": eta[:, 1], "phi": phi[:, 1]})
    invariant_mass = (p1 + p2).mass  # Calculated before use
    
    plt.hist(ak.to_numpy(invariant_mass), bins=60)  # Flattened jagged array
    plt.show()
    
    print(f"Number of events after multiplicity cut: {len(data)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <input_file.root>")
        sys.exit(1)
    main(sys.argv[1])