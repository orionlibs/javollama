package io.github.orionlibs.javollama;

import java.util.function.IntConsumer;
import java.util.stream.IntStream;

final class Parallel
{
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action)
    {
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }
}
