#pragma once

#include "DLLMacros.h"

#ifdef __cplusplus
class IDriftEstimator;
#else //matlab header parsing, so it just looks like a void pointer for 
typedef void IDriftEstimator;
#endif

CDLL_EXPORT IDriftEstimator* DME_CreateInstance(const float* coords_, const float* crlb_, const int* spotFramenum, int numspots,
	float* drift, int framesPerBin, float gradientStep, float maxdrift, int flags, int maxneighbors);

// Drift estimation step. Zero pointers can be passed to status_msg, score, and drift_estimate if not needed
CDLL_EXPORT int DME_Step(IDriftEstimator* estimator, char* status_msg, int status_max_length, float* score, float* drift_estimate);


CDLL_EXPORT void DME_Close(IDriftEstimator* estim);

// included for matlab H parsing
CDLL_EXPORT void Gauss2D_Draw(float* image, int imgw, int imgh, float* spotList, int nspots);
