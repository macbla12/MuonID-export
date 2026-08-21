#ifndef __MuonID_H
#define __MuonID_H
#include "podio/Frame.h"
#include "edm4hep/utils/vector_utils.h"
#include "edm4hep/utils/kinematics.h"
#include "edm4hep/MCParticleCollection.h"
#include "edm4eic/ReconstructedParticleCollection.h"
#include "edm4eic/ClusterCollection.h"
#include "edm4eic/TrackClusterMatchCollection.h"
#include "edm4eic/MCRecoParticleAssociationCollection.h"

// Glowna funkcja klasyfikatora - odpowiednik ElectronID().
// Zwraca P(muon) w [0,1], wybierajac jedna z dwoch sesji ONNX
// (niska / wysoka energia pedu) w zaleznosci od pedu czastki.
double MuonID(edm4eic::ReconstructedParticle rcp, const podio::Frame* event);

// Jednorazowa inicjalizacja obu sesji ONNX Runtime (wolana automatycznie
// przy pierwszym wywolaniu MuonID(), ale mozna tez wywolac recznie wczesniej,
// np. na poczatku makra, zeby "rozgrzac" model przed petla po eventach).
void InitMuonIDSessions();

#endif