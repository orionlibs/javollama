package io.github.orionlibs.javollama.llama;

import io.github.orionlibs.javollama.core.Configuration;
import io.github.orionlibs.javollama.core.ModelLoader;
import io.github.orionlibs.javollama.core.RotaryPositionEmbeddings;
import io.github.orionlibs.javollama.core.Tokenizer;
import io.github.orionlibs.javollama.core.Vocabulary;
import io.github.orionlibs.javollama.core.Weights;
import io.github.orionlibs.javollama.core.gguf.GPTGeneratedUnifiedFormat;
import io.github.orionlibs.javollama.core.tensor.GGUFTensorEntry;
import io.github.orionlibs.javollama.core.utils.Pair;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class LlamaModelLoader extends ModelLoader
{
    private static final String TOKENIZER_LLAMA_3_MODEL = "gpt2";
    private static final String LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";


    public LlamaModelLoader()
    {
        super(TOKENIZER_LLAMA_3_MODEL, LLAMA_3_PATTERN);
    }


    @Override
    public LlamaProcessor loadModel(Path ggufPath, int contextLength) throws IOException
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
        Configuration config = new Configuration(
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
        Pair<float[], float[]> ropeFreqs = RotaryPositionEmbeddings.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta,
                        ropeScaling, scaleFactor, loFreqFactor, hiFreqFactor, oldContextLength);
        float[] ropeFreqsReal = ropeFreqs.first();
        float[] ropeFreqsImag = ropeFreqs.second();
        Map<String, GGUFTensorEntry> tensorEntries = gguf.getTensorEntries();
        Weights qw = new Weights(
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
        return new LlamaProcessor(config, tokenizer, qw);
    }


    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary)
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
}
