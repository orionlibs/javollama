package io.github.orionlibs.javollama.core.utils;

@FunctionalInterface
public interface AggregateFunction
{
    float apply(float acc, float value);
}
