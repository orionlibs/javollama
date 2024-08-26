package io.github.orionlibs.javollama.core.sampler;

import io.github.orionlibs.javollama.core.tensor.FloatTensor;

@FunctionalInterface
public interface Sampler
{
    int sampleToken(FloatTensor logits);


    Sampler ARGMAX = FloatTensor::argmax;
}
