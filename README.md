# MuonID

This directory contains the main implementation of the muon identification function (`MuonID`) within the EPIC analysis framework.

## Directory contents

- **`MuonID.cxx`** — the main `MuonID(...)` function that computes the probability of a muon being identified for a given track and event frame. It does not loop over events or generate histograms (that is the responsibility of the main analysis).
- **`example.cxx`** — example usage of the `MuonID` function.
- **`ToFSim.cxx`** — auxiliary/simulation macro for the ToF detector.
- **`ONNX/`** — exported classifier models used by the `MuonID` function.
- **Output files / plots:** `muID.pdf`, `muID_efficiency.pdf`, `MuonID_Histograms.root` — generated performance and validation plots.

## Requirements and startup

Before running, make sure you are in the **EIC Shell** environment and that the **ONNX Runtime** library environment variables are configured:

```bash
eic-shell
onnx_setup.sh