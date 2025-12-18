import uproot
import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt
import sys

vector.register_awkward()

# User-provided values
PT_CUT = 1.0  # Example cut: PT > 1.0 GeV
ETA_CUT = 2.4  # Example cut: |eta| < 2.4
REQUIRED_COUNT = 2  # Example cut: Exactly 2 particles per event
MASS = 0.10566  # Example mass: Muon
LO = -5.0  # Example range: -5.0 GeV
HI = 5.0  # Example range: 5.0 GeV
DPI = 300  # Example resolution: 300 DPI

def main(input_filename):
    # === LOAD DATA ===
    file = uproot.open(input_filename)
    tree = file["Delphes"]
    # Map user's particle names to Delphes collection
    selected_branches = [
        "Electron/Electron.PT",
        "Electron/Electron.Eta",
        "Electron/Electron.Phi",
        "Electron/Electron.Charge",
    ]
    data = tree.arrays(selected_branches, library="ak")
    pt = data["Electron/Electron.PT"]
    eta = data["Electron/Electron.Eta"]
    phi = data["Electron/Electron.Phi"]
    charge = data["Electron/Electron.Charge"]
    
    n_total = len(pt)
    
    # === RECONSTRUCT 4-MOMENTA ===
    vec = vector.zip({
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "mass": ak.ones_like(pt) * MASS
    })
    
    # === APPLY CUTS IN ORDER ===
    # 1. Object-level cuts
    obj_mask = (pt > PT_CUT) & (np.abs(eta) < ETA_CUT)
    pt = pt[obj_mask]
    eta = eta[obj_mask]
    phi = phi[obj_mask]
    charge = charge[obj_mask]
    vec = vec[obj_mask]
    n_after_object = len(pt)
    
    # 2. Event-level cuts
    n_particles = ak.num(pt)
    event_mask = (n_particles == REQUIRED_COUNT)
    pt = pt[event_mask]
    eta = eta[event_mask]
    phi = phi[event_mask]
    charge = charge[event_mask]
    
    # 3. Additional cuts
    lead_mask = (pt[:, 0] > LEADING_PT_CUT)
    additional_mask = lead_mask & (charge[:, 0] * charge[:, 1] < 0)
    pt = pt[additional_mask]
    eta = eta[additional_mask]
    phi = phi[additional_mask]
    charge = charge[additional_mask]
    n_after_all_cuts = len(pt)
    
    # === PLOTS ===
    # Plot variables mentioned by the user
    plt.figure()
    plt.hist(ak.to_numpy(invariant_mass), bins=N, range=(LO, HI), histtype='step', linewidth=1.5)
    plt.xlabel("Invariant Mass [GeV]")
    plt.ylabel("Events")
    plt.title("Invariant Mass after cuts")
    plt.savefig("mass_after_cuts.png", dpi=DPI)
    
    plt.figure()
    plt.hist(ak.to_numpy(ak.flatten(eta)), bins=N, range=(LO, HI), histtype='step', linewidth=1.5)
    plt.xlabel("Eta")
    plt.ylabel("Particles after cuts")
    plt.title("Eta distribution after cuts")
    plt.savefig("eta_after_cuts.png", dpi=DPI)
    
    # === CUT FLOW SUMMARY ===
    print("=" * 50)
    print("CUT FLOW")
    print("=" * 50)
    print(f"Total events:              {n_total}")
    print(f"After object cuts:         {n_after_object}")
    print(f"After multiplicity cut:    {n_after_multiplicity}")
    print(f"After all cuts:            {n_after_all_cuts}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <input_file.root>")
        sys.exit(1)
    main(sys.argv[1])