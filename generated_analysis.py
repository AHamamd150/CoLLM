import sys
import uproot
import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt

vector.register_awkward()

def find_branch(branches, candidates):
    """Return first matching branch name from candidates list."""
    for c in candidates:
        if c in branches:
            return c
    raise KeyError(f"None of {candidates} found in branches")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <input.root>")
        sys.exit(1)
    
    # Open ROOT file
    file = uproot.open(sys.argv[1])
    tree = file["Delphes"]
    branches = set(tree.keys())
    
    # Find branch names based on selection cuts
    pt_branch = find_branch(branches, ["Electron.PT", "Electron/Electron.PT"])
    eta_branch = find_branch(branches, ["Electron.Eta", "Electron/Electron.Eta"])
    phi_branch = find_branch(branches, ["Electron.Phi", "Electron/Electron.Phi"])
    charge_branch = find_branch(branches, ["Electron.Charge", "Electron/Electron.Charge"])
    
    # Load arrays
    arrays = tree.arrays([pt_branch, eta_branch, phi_branch, charge_branch], library="ak")
    
    # Extract arrays
    electron_pt = arrays[pt_branch]
    electron_eta = arrays[eta_branch]
    electron_phi = arrays[phi_branch]
    electron_charge = arrays[charge_branch]
    
    # Total events before cuts
    n_total = len(electron_pt)
    print(f"Total events: {n_total}")
    
    # Object-level cuts: pT > 20 GeV, |eta| < 2.4
    obj_mask = (electron_pt > 20) & (np.abs(electron_eta) < 2.4)
    electron_pt = electron_pt[obj_mask]
    electron_eta = electron_eta[obj_mask]
    electron_phi = electron_phi[obj_mask]
    electron_charge = electron_charge[obj_mask]
    
    # Multiplicity cut: exactly 2 electrons
    mult_mask = ak.num(electron_pt) == 2
    electron_pt = electron_pt[mult_mask]
    electron_eta = electron_eta[mult_mask]
    electron_phi = electron_phi[mult_mask]
    electron_charge = electron_charge[mult_mask]
    
    n_after_mult = len(electron_pt)
    print(f"After exactly 2 electrons: {n_after_mult}")
    
    # Opposite-sign cut
    os_mask = (electron_charge[:, 0] * electron_charge[:, 1]) < 0
    electron_pt = electron_pt[os_mask]
    electron_eta = electron_eta[os_mask]
    electron_phi = electron_phi[os_mask]
    electron_charge = electron_charge[os_mask]
    
    n_after_os = len(electron_pt)
    print(f"After opposite-sign: {n_after_os}")
    
    # Leading pT cut
    lead_pt_mask = electron_pt[:, 0] > 25
    electron_pt = electron_pt[lead_pt_mask]
    electron_eta = electron_eta[lead_pt_mask]
    electron_phi = electron_phi[lead_pt_mask]
    electron_charge = electron_charge[lead_pt_mask]
    
    n_final = len(electron_pt)
    print(f"After leading pT > 25: {n_final}")
    
    # Build 4-vectors for invariant mass
    electron_mass = ak.ones_like(electron_pt) * 0.000511
    e1 = vector.zip({"pt": electron_pt[:, 0], "eta": electron_eta[:, 0], 
                     "phi": electron_phi[:, 0], "mass": electron_mass[:, 0]})
    e2 = vector.zip({"pt": electron_pt[:, 1], "eta": electron_eta[:, 1], 
                     "phi": electron_phi[:, 1], "mass": electron_mass[:, 1]})
    dielectron = e1 + e2
    dielectron_mass = dielectron.mass
    
    # Z window count
    z_mask = (dielectron_mass > 80) & (dielectron_mass < 100)
    n_z_window = ak.sum(z_mask)
    print(f"Events in Z window (80-100 GeV): {n_z_window}")
    
    # Plot 1: Dielectron mass
    plt.figure(figsize=(8, 6))
    plt.hist(ak.to_numpy(dielectron_mass), bins=60, range=(60, 120), histtype='step')
    plt.xlabel("Dielectron Mass [GeV]")
    plt.ylabel("Events")
    plt.title("Dielectron Invariant Mass")
    plt.savefig("dielectron_mass.png", dpi=150)
    plt.close()
    
    # Plot 2: Leading electron pT
    plt.figure(figsize=(8, 6))
    plt.hist(ak.to_numpy(electron_pt[:, 0]), bins=50, range=(0, 150), histtype='step')
    plt.xlabel("Leading Electron pT [GeV]")
    plt.ylabel("Events")
    plt.title("Leading Electron pT")
    plt.savefig("leading_electron_pt.png", dpi=150)
    plt.close()
    
    # Plot 3: Subleading electron pT
    plt.figure(figsize=(8, 6))
    plt.hist(ak.to_numpy(electron_pt[:, 1]), bins=50, range=(0, 100), histtype='step')
    plt.xlabel("Subleading Electron pT [GeV]")
    plt.ylabel("Events")
    plt.title("Subleading Electron pT")
    plt.savefig("subleading_electron_pt.png", dpi=150)
    plt.close()
    
    # Plot 4: Electron eta (all electrons, flattened)
    all_eta = ak.flatten(electron_eta)
    plt.figure(figsize=(8, 6))
    plt.hist(ak.to_numpy(all_eta), bins=50, range=(-2.5, 2.5), histtype='step')
    plt.xlabel("Electron Eta")
    plt.ylabel("Electrons")
    plt.title("Electron Eta Distribution")
    plt.savefig("electron_eta.png", dpi=150)
    plt.close()
    
    print("Plots saved successfully.")

if __name__ == "__main__":
    main()