# MuonID

Ten katalog zawiera główną implementację funkcji identyfikacji mionów (`MuonID`) w ramach analiz EPIC.

## Zawartość katalogu

- **`MuonID.cxx`** — główna funkcja `MuonID(...)` obliczająca prawdopodobieństwo identyfikacji mionu dla zadanej trajektorii (*track*) oraz ramki zdarzenia. Nie pętluje po zdarzeniach ani nie generuje histogramów (to zadanie analizy głównej).
- **`example.cxx`** — przykładowe użycie funkcji `MuonID`.
- **`ToFSim.cxx`** — makro pomocnicze/symulacyjne dla detektora ToF.
- **`ONNX/`** — wyeksportowane modele klasyfikatorów używane przez funkcję `MuonID`.
- **Pliki wyjściowe / wykresy:** `muID.pdf`, `muID_efficiency.pdf`, `MuonID_Histograms.root` — wygenerowane wykresy wydajności i walidacji.

## Wymagania i uruchomienie

Przed uruchomieniem upewnij się, że znajdujesz się w środowisku **EIC Shell** oraz że zmienne dla biblioteki **ONNX Runtime** są skonfigurowane:

```bash
source /usr/local/eic/eic-shell
source ../onnx_setup.sh