package io.github.orionlibs.javollama;

@FunctionalInterface
public interface MapWithIndexFunction
{
    float apply(float value, int index);
}
