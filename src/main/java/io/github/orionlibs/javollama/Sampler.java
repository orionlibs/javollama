package io.github.orionlibs.javollama;

@FunctionalInterface
public interface Sampler
{
    int sampleToken(FloatTensor logits);


    Sampler ARGMAX = FloatTensor::argmax;
}
