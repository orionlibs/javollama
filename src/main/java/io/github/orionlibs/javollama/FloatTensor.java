package io.github.orionlibs.javollama;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Over-simplified, shapeless, float tensor.
 * <p>
 * Not a strict tensor, but rather just a sequence of floats, not required to be backed by memory
 * e.g. can represent a sequence of quantized floats.
 */
public abstract class FloatTensor
{
    static final ValueLayout.OfFloat JAVA_FLOAT_LE = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN);
    static final ValueLayout.OfShort JAVA_SHORT_LE = ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);
    static final boolean USE_VECTOR_API = Boolean.parseBoolean(System.getProperty("llama.VectorAPI", "true"));
    // Preferred vector size for the fast multiplication routines.
    // (Apple Silicon) NEON only supports up-to 128bit vectors.
    static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED.vectorBitSize() == 128 ? FloatVector.SPECIES_128 : FloatVector.SPECIES_256;


    public abstract int size();


    abstract float getFloat(int index);


    abstract void setFloat(int index, float value);


    abstract FloatVector getFloatVector(VectorSpecies<Float> species, int offset);


    abstract GGMLType type();


    public static int numberOfElements(int... dimensions)
    {
        assert Arrays.stream(dimensions).allMatch(i -> i > 0);
        return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
    }


    static float scalarDot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size)
    {
        float result = 0f;
        for(int j = 0; j < size; j++)
        {
            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
        }
        return result;
    }


    float dot(int thisOffset, FloatTensor that, int thatOffset, int size)
    {
        return scalarDot(this, thisOffset, that, thatOffset, size);
    }


    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1)
    {
        Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(i * dim1, that, 0, dim1)));
    }


    float reduce(int thisOffset, int size, float seed, AggregateFunction reduce)
    {
        float result = seed;
        for(int i = 0; i < size; ++i)
        {
            result = reduce.apply(result, getFloat(thisOffset + i));
        }
        return result;
    }


    float sum(int thisOffset, int size)
    {
        return reduce(thisOffset, size, 0f, Float::sum);
    }


    float max(int thisOffset, int size)
    {
        return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Float::max);
    }


    void copyTo(int thisOffset, FloatTensor that, int thatOffset, int size)
    {
        that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
    }


    int argmax(int thisOffset, int size)
    {
        assert size > 0;
        int maxIndex = thisOffset;
        float maxValue = this.getFloat(maxIndex);
        int endIndex = thisOffset + size;
        for(int i = thisOffset; i < endIndex; ++i)
        {
            float f = this.getFloat(i);
            if(f > maxValue)
            {
                maxValue = f;
                maxIndex = i;
            }
        }
        return maxIndex;
    }


    int argmax()
    {
        return argmax(0, size());
    }


    FloatTensor mapInPlace(int thisOffset, int size, MapFunction mapFunction)
    {
        int endIndex = thisOffset + size;
        for(int i = thisOffset; i < endIndex; ++i)
        {
            setFloat(i, mapFunction.apply(getFloat(i)));
        }
        return this;
    }


    FloatTensor mapInPlace(MapFunction mapFunction)
    {
        return mapInPlace(0, size(), mapFunction);
    }


    FloatTensor mapWithIndexInPlace(int thisOffset, int size, MapWithIndexFunction mapWithIndexFunction)
    {
        int endOffset = thisOffset + size;
        for(int i = thisOffset; i < endOffset; ++i)
        {
            setFloat(i, mapWithIndexFunction.apply(getFloat(i), i));
        }
        return this;
    }


    FloatTensor addInPlace(int thisOffset, FloatTensor that, int thatOffset, int size)
    {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value + that.getFloat(index - thisOffset + thatOffset));
    }


    FloatTensor addInPlace(FloatTensor that)
    {
        return addInPlace(0, that, 0, size());
    }


    FloatTensor multiplyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size)
    {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value * that.getFloat(index - thisOffset + thatOffset));
    }


    FloatTensor multiplyInPlace(FloatTensor that)
    {
        return multiplyInPlace(0, that, 0, size());
    }


    public FloatTensor divideInPlace(int thisOffset, int size, float value)
    {
        return mapInPlace(thisOffset, size, f -> f / value);
    }


    FloatTensor fillInPlace(int thisOffset, int size, float value)
    {
        return mapInPlace(thisOffset, size, unused -> value);
    }


    public FloatTensor softmaxInPlace(int thisOffset, int size)
    {
        // find max value (for numerical stability)
        float maxVal = max(thisOffset, size);
        // exp and sum
        mapInPlace(thisOffset, size, f -> (float)Math.exp(f - maxVal));
        float sum = sum(thisOffset, size);
        // normalize
        return divideInPlace(thisOffset, size, sum);
    }


    FloatTensor saxpyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size, float a)
    {
        // this[thatOffset ... thatOffset + size) = a * that[thatOffset ... thatOffset + size) + this[thisOffset ... thisOffset + size)
        for(int i = 0; i < size; ++i)
        {
            setFloat(thisOffset + i, a * that.getFloat(thatOffset + i) + this.getFloat(thisOffset + i));
        }
        return this;
    }
}
