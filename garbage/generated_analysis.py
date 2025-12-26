import sys
import uproot
import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt

vector.register_awkward()

def find_branch(branches, candidates):
    for c in candidates:
        if c in branches:
            return c
    raise KeyError(f"None of {candidates} found in branches")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <input.root>")
        sys.exit(1)

    file = uproot.open(sys.argv[1])
    tree = file["Delphes"]
    branches = set(tree.keys())

    # Find branches
    mu_pt_br = find_branch(branches, ["Muon.PT", "Muon/Muon.PT"])
    mu_eta_br = find_branch(branches, ["Muon.Eta", "Muon/Muon.Eta"])
    mu_charge_br = find_branch(branches, ["Muon.Charge", "Muon/Muon.Charge"])

    # Load arrays
    arrays = tree.arrays([mu_pt_br, mu_eta_br, mu_charge_br], library="ak")

    muon_pt = arrays[mu_pt_br]
    muon_eta = arrays[mu_eta_br]
    muon_charge = arrays[mu_charge_br]

    # Total events BEFORE cuts
    n_total = len(muon_pt)
    print(f"Total events: {n_total}")

    # Muon cuts: pT > 20, |eta| < 2.4
    mu_mask = (muon_pt > 20) & (np.abs(muon_eta) < 2.4)
    muon_pt = muon_pt[mu_mask]
    muon_eta = muon_eta[mu_mask]
    muon_charge = muon_charge[mu_mask]

    # At least 2 muons
    mask = ak.num(muon_pt) >= 2
    muon_pt = muon_pt[mask]
    muon_eta = muon_eta[mask]
    muon_charge = muon_charge[mask]
    n_after_muons = len(muon_pt)
    print(f"After muon selection: {n_after_muons}")

    # Exactly 2 muons
    mask = ak.num(muon_pt) == 2
    muon_pt = muon_pt[mask]
    muon_eta = muon_eta[mask]
    muon_charge = muon_charge[mask]
    n_after_exactly_two = len(muon_pt)
    print(f"After exactly 2 muons: {n_after_exactly_two}")

    # Opposite-sign muon pair
    opposite_sign_mask = muon_charge[:, 0] * muon_charge[:, 1] < 0
    muon_pt = muon_pt[opposite_sign_mask]
    muon_eta = muon_eta[opposite_sign_mask]
    muon_charge = muon_charge[opposite_sign_mask]
    n_after_opposite_sign = len(muon_pt)
    print(f"After opposite-sign requirement: {n_after_opposite_sign}")

    # Calculate dimuon invariant mass
    mu1_vec = vector.zip({"pt": muon_pt[:, 0], "eta": muon_eta[:, 0], "phi": muon_phi[:, 0], "mass": ak.ones_like(muon_pt[:, 0]) * 0.10566})
    mu2_vec = vector.zip({"pt": muon_pt[:, 1], "eta": muon_eta[:, 1], "phi": muon_phi[:, 1], "mass": ak.ones_like(muon_pt[:, 1]) * 0.10566})
    dimuon_mass = (mu1_vec + mu2_vec).mass

    # Events in Z mass window (80-100 GeV)
    z_window_mask = (dimuon_mass > 80) & (dimuon_mass < 100)
    dimuon_mass = dimuon_mass[z_window_mask]
    n_in_z_window = len(dimuon_mass)
    print(f"Events in Z mass window: {n_in_z_window}")

    # Plot dimuon invariant mass
    plt.figure(figsize=(8, 6))
    plt.hist(ak.to_numpy(ak.flatten(dimuon_mass)), bins=50, range=(60, 120), histtype='step')
    plt.xlabel("Dimuon Invariant Mass [GeV]")
    plt.ylabel("Events")
    plt.title("Dimuon Invariant Mass")
    plt.savefig("dimuon_mass.png", dpi=150)
    plt.close()

    # Plot leading muon pT
    plt.figure(figsize=(8, 6))
    plt.hist(ak.to_numpy(ak.flatten(muon_pt[:, 0])), bins=50, range=(0, 150), histtype='step')
    plt.xlabel("Leading Muon pT [GeV]")
    plt.ylabel("Muons")
    plt.title("Leading Muon pT")
    plt.savefig("leading_muon_pt.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
