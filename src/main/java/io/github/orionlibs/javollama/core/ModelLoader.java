package io.github.orionlibs.javollama.core;

import io.github.orionlibs.javollama.core.gguf.GGUFType;
import io.github.orionlibs.javollama.core.tensor.FloatTensor;
import io.github.orionlibs.javollama.core.tensor.GGUFTensorEntry;
import io.github.orionlibs.javollama.core.tensor.Q4_0FloatTensor;
import io.github.orionlibs.javollama.core.tensor.Q8_0FloatTensor;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.IntFunction;

public abstract class ModelLoader
{
    protected String TOKENIZER_MODEL;
    protected String PATTERN;


    public ModelLoader(String TOKENIZER_MODEL, String PATTERN)
    {
        this.TOKENIZER_MODEL = TOKENIZER_MODEL;
        this.PATTERN = PATTERN;
    }


    public Vocabulary loadVocabulary(Map<String, Object> metadata)
    {
        String model = (String)metadata.get("tokenizer.ggml.model");
        if(!TOKENIZER_MODEL.equals(model))
        {
            throw new IllegalArgumentException("expected " + TOKENIZER_MODEL + " but found " + model);
        }
        String[] tokens = (String[])metadata.get("tokenizer.ggml.tokens");
        return new Vocabulary(tokens, null);
    }


    public abstract LLMProcessor loadModel(Path ggufPath, int contextLength) throws IOException;


    protected abstract Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary);


    public static FloatTensor loadQuantized(GGUFTensorEntry entry)
    {
        GGUFType ggmlType = entry.ggmlType();
        return switch(ggmlType)
        {
            //case F32 -> new F32FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q8_0 -> new Q8_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }


    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGUFTensorEntry> getTensorEntry)
    {
        FloatTensor[] array = new FloatTensor[size];
        for(int i = 0; i < size; i++)
        {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }


    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGUFTensorEntry> getTensorEntry)
    {
        FloatBuffer[] array = new FloatBuffer[size];
        for(int i = 0; i < size; i++)
        {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }


    public static FloatBuffer toFloatBuffer(GGUFTensorEntry tensorEntry)
    {
        GGUFType ggmlType = tensorEntry.ggmlType();
        return switch(ggmlType)
        {
            case F32 -> tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }
}
