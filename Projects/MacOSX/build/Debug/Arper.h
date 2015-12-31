#ifndef TINYRNN_STANDALONE_GUARD_7cfe5fe06de146adba68aadf55423c5bd2d2a369d6b0eabc0d1bb2a7b6fcbc36
#define TINYRNN_STANDALONE_GUARD_7cfe5fe06de146adba68aadf55423c5bd2d2a369d6b0eabc0d1bb2a7b6fcbc36

extern float kMemory[];
const int kMemorySize = 6235;

extern float kOutputs[];
const int kOutputsSize = 4;

void ArperFeed(const float *input);
void ArperTrain(const float rate, const float *target);

#endif //TINYRNN_STANDALONE_GUARD_7cfe5fe06de146adba68aadf55423c5bd2d2a369d6b0eabc0d1bb2a7b6fcbc36
