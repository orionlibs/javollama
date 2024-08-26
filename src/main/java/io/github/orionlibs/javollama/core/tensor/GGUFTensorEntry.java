package io.github.orionlibs.javollama.core.tensor;

import io.github.orionlibs.javollama.core.gguf.GGUFType;
import java.lang.foreign.MemorySegment;

public record GGUFTensorEntry(MemorySegment mappedFile, String name, GGUFType ggmlType, int[] shape,
                              MemorySegment memorySegment)
{
}
