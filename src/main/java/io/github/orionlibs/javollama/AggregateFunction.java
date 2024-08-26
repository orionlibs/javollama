package io.github.orionlibs.javollama;

@FunctionalInterface
public interface AggregateFunction
{
    float apply(float acc, float value);
}
