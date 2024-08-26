JavOllama

Librarified and slightly refactored version of the [llama3.java](https://github.com/mukel/llama3.java) project. It requires JRE21+. It works with Llama and other LLMs. For more info, please check the link above of the llama3.java project. In order to run this project, you need a copy of a GGUF model like this one: https://huggingface.co/mukel/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_0.gguf

The Maven coordinates of this library are:

```xml
<dependency>
    <groupId>io.github.orionlibs</groupId>
    <artifactId>javollama</artifactId>
    <version>1.0.1</version>
</dependency>
```

add this line to IntelliJ's compiler settings shared build VM options: --enable-preview --add-modules jdk.incubator.vector

You can see how to use the library by looking at the LLMTest.java class
