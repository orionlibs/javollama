package io.github.orionlibs.javollama;

import java.util.stream.Stream;

public final class State
{
    // current wave of activations
    public final FloatTensor x; // activation at current time stamp (dim,)
    public final FloatTensor xb; // same, but inside a residual branch (dim,)
    public final FloatTensor xb2; // an additional buffer just for convenience (dim,)
    public final FloatTensor hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    public final FloatTensor hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    public final FloatTensor q; // query (dim,)
    public final FloatTensor k; // key (dim,)
    public final FloatTensor v; // value (dim,)
    public final FloatTensor att; // buffer for scores/attention values (n_heads, seq_len)
    public final FloatTensor logits; // output logits
    // kv cache
    public final FloatTensor[] keyCache;   // (n_layer, seq_len, kv_dim)
    public final FloatTensor[] valueCache; // (n_layer, seq_len, kv_dim)
    public int latestToken;


    State(Configuration config)
    {
        this.x = ArrayFloatTensor.allocate(config.dim);
        this.xb = ArrayFloatTensor.allocate(config.dim);
        this.xb2 = ArrayFloatTensor.allocate(config.dim);
        this.hb = ArrayFloatTensor.allocate(config.hiddenDim);
        this.hb2 = ArrayFloatTensor.allocate(config.hiddenDim);
        this.q = ArrayFloatTensor.allocate(config.dim);
        this.k = ArrayFloatTensor.allocate(config.dim);
        this.v = ArrayFloatTensor.allocate(config.dim);
        this.att = ArrayFloatTensor.allocate(config.numberOfHeads, config.contextLength);
        this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        this.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
        this.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
    }
}
