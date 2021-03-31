// Drift estimation using entropy minimization (DME)
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "DLLMacros.h"
#include "Vector.h"


CDLL_EXPORT void CrossCorrelationDriftEstimate(const Vector3f* xyI, const int *spotFrameNum, int numspots, const Int2* framePairs, int numframepairs,
	Vector2f* drift, int width, int height, float maxDrift);

CDLL_EXPORT int GaussianOverlapDriftEstimate(const float* xy, const int* spotFramenum, int numspots,
	const float* sigma, int maxiterations, float* driftXY, int framesPerBin, float gradientStep, float maxdrift, float* scores, int flags, int (*progcb)(int iteration, const char* info));


