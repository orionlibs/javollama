package io.github.orionlibs.javollama.core.gguf;

import io.github.orionlibs.javollama.core.utils.Float16;

public enum GGUFType
{
    F32(Float.BYTES),
    F16(Float16.BYTES),
    Q4_0(Float16.BYTES + 16 * Byte.BYTES, 32),
    Q4_1(2 * Float16.BYTES + 16 * Byte.BYTES, 32),
    UNSUPPORTED_Q4_2(Integer.MAX_VALUE), // support has been removed
    UNSUPPORTED_Q4_3(Integer.MAX_VALUE), // support has been removed
    Q5_0(Integer.MAX_VALUE),
    Q5_1(Integer.MAX_VALUE),
    Q8_0(Float16.BYTES + 32 * Byte.BYTES, 32),
    Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, 32),
    // k-quantizations
    Q2_K(Integer.MAX_VALUE),
    Q3_K(Integer.MAX_VALUE),
    Q4_K(2 * Float16.BYTES + ((GGUFType.QK_K / 16) / 8 * 6) + GGUFType.QK_K / 2, GGUFType.QK_K),
    Q5_K(2 * Float16.BYTES + ((GGUFType.QK_K / 16) / 8 * 6) + GGUFType.QK_K / 8 + GGUFType.QK_K / 2, GGUFType.QK_K),
    Q6_K(GGUFType.QK_K / 2 + GGUFType.QK_K / 4 + GGUFType.QK_K / 16 + Float16.BYTES, GGUFType.QK_K),
    Q8_K(Integer.MAX_VALUE),
    I8(Byte.BYTES),
    I16(Short.BYTES),
    I32(Integer.BYTES);
    private static final GGUFType[] VALUES = values();
    private final int typeSize;
    private final int blockSize;


    public int getTypeSize()
    {
        return typeSize;
    }


    public int getBlockSize()
    {
        return blockSize;
    }


    public static GGUFType fromId(int id)
    {
        return VALUES[id];
    }


    GGUFType(int typeSize)
    {
        this(typeSize, 1);
    }


    public long byteSizeFor(int numberOfElements)
    {
        long t = numberOfElements * (long)getTypeSize();
        assert t % getBlockSize() == 0;
        return Math.toIntExact(t / getBlockSize());
    }


    public static final int QK_K = 256; // or 64?


    GGUFType(int typeSize, int blockSize)
    {
        assert blockSize > 0;
        assert typeSize > 0;
        assert isPowerOf2(blockSize);
        this.typeSize = typeSize;
        this.blockSize = blockSize;
    }


    private static boolean isPowerOf2(int n)
    {
        return n > 0 && (n & (n - 1)) == 0;
    }
}
