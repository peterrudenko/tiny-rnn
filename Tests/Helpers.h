
#ifndef HELPERS_H_INCLUDED
#define HELPERS_H_INCLUDED

#define RANDOM(x, y) Helpers::random(x, y)
#define RANDOMNAME() Helpers::randomName()

#include "Uuid.h"
#include <random>

struct Helpers
{
    static std::random_device &device()
    {
        static std::random_device randomDevice;
        return randomDevice;
    }
    
    static std::mt19937 &twister()
    {
        static std::mt19937 mt19937(Helpers::device()());
        return mt19937;
    }
    
    static int random(int from, int to)
    {
        std::uniform_int_distribution<int> distribution(from, to);
        return distribution(Helpers::twister());
    }
    
    static double random(double from, double to)
    {
        std::uniform_real_distribution<double> distribution(from, to);
        return distribution(Helpers::twister());
    }
    
    static std::string randomName()
    {
        return TinyRNN::Uuid::generateIsoUuid();
    }
};

#endif // HELPERS_H_INCLUDED
