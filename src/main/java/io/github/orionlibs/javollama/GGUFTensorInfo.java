package io.github.orionlibs.javollama;

public record GGUFTensorInfo(String name, int[] dimensions, GGMLType ggmlType, long offset)
{
}
