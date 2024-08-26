package io.github.orionlibs.javollama.core.utils;

@FunctionalInterface
public interface MapWithIndexFunction
{
    float apply(float value, int index);
}
