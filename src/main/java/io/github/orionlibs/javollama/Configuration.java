package io.github.orionlibs.javollama;

public final class Configuration
{
    public final int dim; // transformer dimension
    public final int hiddenDim; // for ffn layers
    public final int numberOfLayers; // number of layers
    public final int numberOfHeads; // number of query heads
    public final int numberOfKeyValueHeads; // number of key/value heads (can be < query heads because of multiquery)
    public final int vocabularySize; // vocabulary size, usually 256 (byte-level)
    public final int contextLength; // max sequence length
    public final boolean sharedWeights;
    public final float rmsNormEps;
    public final float ropeTheta;
    public final int headSize;


    public Configuration(int dim, int hiddenDim, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads, int vocabularySize, int contextLength, boolean sharedWeights, float rmsNormEps, float ropeTheta)
    {
        this.dim = dim;
        this.hiddenDim = hiddenDim;
        this.numberOfLayers = numberOfLayers;
        this.numberOfHeads = numberOfHeads;
        this.numberOfKeyValueHeads = numberOfKeyValueHeads;
        this.vocabularySize = vocabularySize;
        this.contextLength = contextLength;
        this.sharedWeights = sharedWeights;
        this.rmsNormEps = rmsNormEps;
        this.ropeTheta = ropeTheta;
        this.headSize = dim / numberOfHeads;
    }
}
