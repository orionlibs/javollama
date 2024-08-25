package io.github.orionlibs.javollama;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public class Llama31
{
    public static void main(String[] args) throws IOException
    {
        Options options = Options.parseOptions(args);
        Llama model = ModelLoader.loadModel(options.modelPath(), options.maxTokens());
        Sampler sampler = selectSampler(model.configuration().vocabularySize, options.temperature(), options.topp(), options.seed());
        if(options.interactive())
        {
            runInteractive(model, sampler, options);
        }
        else
        {
            runInstructOnce(model, sampler, options);
        }
    }


    static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed)
    {
        Sampler sampler;
        if(temperature == 0.0f)
        {
            // greedy argmax sampling: take the token with the highest probability
            sampler = Sampler.ARGMAX;
        }
        else
        {
            // we sample from this distribution to get the next token
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
            Sampler innerSampler;
            if(topp <= 0 || topp >= 1)
            {
                // simply sample from the predicted probability distribution
                innerSampler = new CategoricalSampler(rng);
            }
            else
            {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }
            sampler = logits -> {
                // apply the temperature to the logits
                logits.divideInPlace(0, logits.size(), temperature);
                // apply softmax to the logits to get the probabilities for next token
                logits.softmaxInPlace(0, logits.size());
                return innerSampler.sampleToken(logits);
            };
        }
        return sampler;
    }


    static void runInteractive(Llama model, Sampler sampler, Options options)
    {
        Llama.State state = null;
        List<Integer> conversationTokens = new ArrayList<>();
        ChatFormat chatFormat = new ChatFormat(model.tokenizer());
        conversationTokens.add(chatFormat.beginOfText);
        if(options.systemPrompt() != null)
        {
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        int startPosition = 0;
        Scanner in = new Scanner(System.in);
        while(true)
        {
            System.out.print("> ");
            System.out.flush();
            String userText = in.nextLine();
            if(List.of("quit", "exit").contains(userText))
            {
                break;
            }
            if(state == null)
            {
                state = model.createNewState();
            }
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, userText)));
            conversationTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            Set<Integer> stopTokens = chatFormat.getStopTokens();
            List<Integer> responseTokens = Llama.generateTokens(model, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
                if(options.stream())
                {
                    if(!model.tokenizer().isSpecialToken(token))
                    {
                        System.out.print(model.tokenizer().decode(List.of(token)));
                    }
                }
            });
            // Include stop token in the prompt history, but not in the response displayed to the user.
            conversationTokens.addAll(responseTokens);
            startPosition = conversationTokens.size();
            Integer stopToken = null;
            if(!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast()))
            {
                stopToken = responseTokens.getLast();
                responseTokens.removeLast();
            }
            if(!options.stream())
            {
                String responseText = model.tokenizer().decode(responseTokens);
                System.out.println(responseText);
            }
            if(stopToken == null)
            {
                System.err.println("Ran out of context length...");
                break;
            }
        }
    }


    static void runInstructOnce(Llama model, Sampler sampler, Options options)
    {
        Llama.State state = model.createNewState();
        ChatFormat chatFormat = new ChatFormat(model.tokenizer());
        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(chatFormat.beginOfText);
        if(options.systemPrompt() != null)
        {
            promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        Set<Integer> stopTokens = chatFormat.getStopTokens();
        List<Integer> responseTokens = Llama.generateTokens(model, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
            if(options.stream())
            {
                if(!model.tokenizer().isSpecialToken(token))
                {
                    System.out.print(model.tokenizer().decode(List.of(token)));
                }
            }
        });
        if(!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast()))
        {
            responseTokens.removeLast();
        }
        if(!options.stream())
        {
            String responseText = model.tokenizer().decode(responseTokens);
            System.out.println(responseText);
        }
    }
}