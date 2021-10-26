#pragma once

#include "DLLMacros.h"

#ifdef __cplusplus
class IDriftEstimator;
#define STRUCT
#else //matlab header parsing
typedef struct IDriftEstimator;
#define STRUCT struct
#endif

CDLL_EXPORT STRUCT IDriftEstimator* DME_CreateInstance(const float* coords_, const float* crlb_, const int* spotFramenum, int numspots,
	float* drift, int framesPerBin, float gradientStep, float maxdrift, int flags, int maxneighbors);

// Drift estimation step. Zero pointers can be passed to status_msg, score, and drift_estimate if not needed
CDLL_EXPORT int DME_Step(STRUCT IDriftEstimator* estimator, char* status_msg, int status_max_length, float* score, float* drift_estimate);


CDLL_EXPORT void DME_Close(STRUCT IDriftEstimator* estim);
