package io.github.orionlibs.javollama;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

final class ModelLoader
{
    private static final String TOKENIZER_LLAMA_3_MODEL = "gpt2";
    private static final String LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";


    private static Vocabulary loadVocabulary(Map<String, Object> metadata)
    {
        String model = (String)metadata.get("tokenizer.ggml.model");
        if(!TOKENIZER_LLAMA_3_MODEL.equals(model))
        {
            throw new IllegalArgumentException("expected " + TOKENIZER_LLAMA_3_MODEL + " but found " + model);
        }
        String[] tokens = (String[])metadata.get("tokenizer.ggml.tokens");
        return new Vocabulary(tokens, null);
    }


    public static Llama loadModel(Path ggufPath, int contextLength) throws IOException
    {
        try(var ignored = Timer.log("Load LlaMa model"))
        {
            GPTGeneratedUnifiedFormat gguf = GPTGeneratedUnifiedFormat.loadModel(ggufPath);
            Map<String, Object> metadata = gguf.getMetadata();
            Vocabulary vocabulary = loadVocabulary(metadata);
            Tokenizer tokenizer = createTokenizer(metadata, vocabulary);
            int modelContextLength = (int)metadata.get("llama.context_length");
            if(contextLength < 0 || modelContextLength < contextLength)
            {
                contextLength = modelContextLength;
            }
            Llama.Configuration config = new Llama.Configuration(
                            (int)metadata.get("llama.embedding_length"),
                            (int)metadata.get("llama.feed_forward_length"),
                            (int)metadata.get("llama.block_count"),
                            (int)metadata.get("llama.attention.head_count"),
                            metadata.containsKey("llama.attention.head_count_kv")
                                            ? (int)metadata.get("llama.attention.head_count_kv")
                                            : (int)metadata.get("llama.attention.head_count"),
                            vocabulary.size(),
                            contextLength,
                            false,
                            (float)metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f),
                            (float)metadata.getOrDefault("llama.rope.freq_base", 10000f)
            );
            boolean ropeScaling = "Meta-Llama-3.1".equals(metadata.get("general.basename"));
            float scaleFactor = 8;
            float loFreqFactor = 1;
            float hiFreqFactor = 3;
            int oldContextLength = 8192;
            Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta,
                            ropeScaling, scaleFactor, loFreqFactor, hiFreqFactor, oldContextLength);
            float[] ropeFreqsReal = ropeFreqs.first();
            float[] ropeFreqsImag = ropeFreqs.second();
            Map<String, GGMLTensorEntry> tensorEntries = gguf.getTensorEntries();
            Llama.Weights qw = new Llama.Weights(
                            loadQuantized(tensorEntries.get("token_embd.weight")),
                            loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                            loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                            loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                            loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                            loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                            loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                            loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")), // w1
                            loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // w2
                            loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), // w3
                            toFloatBuffer(tensorEntries.get("output_norm.weight")),
                            FloatBuffer.wrap(ropeFreqsReal),
                            FloatBuffer.wrap(ropeFreqsImag),
                            loadQuantized(tensorEntries.get("output.weight"))
            );
            return new Llama(config, tokenizer, qw);
        }
    }


    private static Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary)
    {
        String[] mergeLines = (String[])metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
                        .map(line -> line.split(" "))
                        .map(parts ->
                                        new Pair<>(
                                                        vocabulary.getIndex(parts[0]).orElseThrow(),
                                                        vocabulary.getIndex(parts[1]).orElseThrow())
                        ).toList();
        int allTokens = vocabulary.size();
        int baseTokens = 128000; // assume all tokens after the base ones are special.
        int reservedSpecialTokens = allTokens - baseTokens;
        List<String> specialTokensList = Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();
        assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());
        Map<String, Integer> specialTokens =
                        IntStream.range(0, specialTokensList.size())
                                        .boxed()
                                        .collect(Collectors.toMap(
                                                        i -> specialTokensList.get(i),
                                                        i -> baseTokens + i)
                                        );
        return new Tokenizer(vocabulary, merges, LLAMA_3_PATTERN, specialTokens);
    }


    public static FloatTensor loadQuantized(GGMLTensorEntry entry)
    {
        GGMLType ggmlType = entry.ggmlType();
        return switch(ggmlType)
        {
            //case F32 -> new F32FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q8_0 -> new Q8_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }


    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry)
    {
        FloatTensor[] array = new FloatTensor[size];
        for(int i = 0; i < size; i++)
        {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }


    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry)
    {
        FloatBuffer[] array = new FloatBuffer[size];
        for(int i = 0; i < size; i++)
        {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }


    public static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry)
    {
        GGMLType ggmlType = tensorEntry.ggmlType();
        return switch(ggmlType)
        {
            case F32 -> tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }
}
