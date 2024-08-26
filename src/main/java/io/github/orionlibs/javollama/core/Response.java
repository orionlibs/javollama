package io.github.orionlibs.javollama.core;

import java.util.ArrayList;
import java.util.List;

public class Response
{
    private StringBuilder content;
    private String statsFormatted;
    private List<Integer> responseTokens;
    private double tokenGenerationRate;
    private int numberOfTokensGenerated;


    public Response(int maxTokens)
    {
        this.content = new StringBuilder();
        this.responseTokens = new ArrayList<>(maxTokens);
    }


    public void appendContent(String newContent)
    {
        content.append(newContent);
    }


    public void addResponseToken(int nextToken)
    {
        this.responseTokens.add(nextToken);
    }


    public String getContent()
    {
        return content.toString();
    }


    public String getStatsFormatted()
    {
        return this.statsFormatted;
    }


    public void setStatsFormatted(String statsFormatted)
    {
        this.statsFormatted = statsFormatted;
    }


    public List<Integer> getResponseTokens()
    {
        return this.responseTokens;
    }


    public double getTokenGenerationRate()
    {
        return tokenGenerationRate;
    }


    public void setTokenGenerationRate(double tokenGenerationRate)
    {
        this.tokenGenerationRate = tokenGenerationRate;
    }


    public int getNumberOfTokensGenerated()
    {
        return numberOfTokensGenerated;
    }


    public void setNumberOfTokensGenerated(int numberOfTokensGenerated)
    {
        this.numberOfTokensGenerated = numberOfTokensGenerated;
    }
}
