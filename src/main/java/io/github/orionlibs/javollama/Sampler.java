package io.github.orionlibs.javollama;

@FunctionalInterface
interface Sampler
{
    int sampleToken(FloatTensor logits);


    Sampler ARGMAX = FloatTensor::argmax;
}
