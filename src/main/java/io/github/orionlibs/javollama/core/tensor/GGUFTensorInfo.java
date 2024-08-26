package io.github.orionlibs.javollama.core.tensor;

import io.github.orionlibs.javollama.core.gguf.GGUFType;

public record GGUFTensorInfo(String name, int[] dimensions, GGUFType ggmlType, long offset)
{
}
