package io.github.orionlibs.javollama.core.gguf;

import io.github.orionlibs.javollama.core.tensor.FloatTensor;
import io.github.orionlibs.javollama.core.tensor.GGUFTensorEntry;
import io.github.orionlibs.javollama.core.tensor.GGUFTensorInfo;
import io.github.orionlibs.javollama.core.utils.MetadataValueType;
import io.github.orionlibs.javollama.core.utils.Pair;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class GPTGeneratedUnifiedFormat
{
    private static final int GGUF_MAGIC = 0x46554747;
    private static final int DEFAULT_ALIGNMENT = 32; // must be a power of 2
    private static final List<Integer> SUPPORTED_GGUF_VERSIONS = List.of(2, 3);
    private int magic;
    private int version;
    private int tensorCount; // uint64_t
    private int alignment;
    private int metadata_kv_count; // uint64_t
    private Map<String, Object> metadata;
    private Map<String, GGUFTensorInfo> tensorInfos;
    private long tensorDataOffset;
    private MemorySegment tensorData; // memory mapped tensor data
    private Map<String, GGUFTensorEntry> tensorEntries;


    public Map<String, Object> getMetadata()
    {
        return metadata;
    }


    private final ByteBuffer BB_1 = ByteBuffer.allocate(Byte.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_2 = ByteBuffer.allocate(Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_4 = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_8 = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);


    public Map<String, GGUFTensorEntry> getTensorEntries()
    {
        return tensorEntries;
    }


    public static GPTGeneratedUnifiedFormat loadModel(Path modelPath) throws IOException
    {
        try(FileChannel fileChannel = FileChannel.open(modelPath))
        {
            GPTGeneratedUnifiedFormat gguf = new GPTGeneratedUnifiedFormat();
            gguf.loadModelImpl(fileChannel);
            return gguf;
        }
    }


    private void loadModelImpl(FileChannel fileChannel) throws IOException
    {
        // The header of the file.
        readHeader(fileChannel); // gguf_header_t header;
        // Tensor infos, which can be used to locate the tensor data.
        // gguf_tensor_info_t tensor_infos[header.tensor_count];
        this.tensorInfos = HashMap.newHashMap(tensorCount);
        for(int i = 0; i < tensorCount; ++i)
        {
            GGUFTensorInfo ti = readTensorInfo(fileChannel);
            assert !tensorInfos.containsKey(ti.name());
            tensorInfos.put(ti.name(), ti);
        }
        // Padding to the nearest multiple of `ALIGNMENT`.
        // uint8_t _padding[ALIGNMENT - (sizeof(header + tensor_infos) % ALIGNMENT)];
        //long _padding = -fileChannel.position() & (ALIGNMENT - 1);
        long _padding = getAlignment() - (fileChannel.position() % getAlignment());
        fileChannel.position(fileChannel.position() + _padding);
        // Tensor data.
        //
        // This is arbitrary binary data corresponding to the weights of the model. This data should be close
        // or identical to the data in the original model file, but may be different due to quantization or
        // other optimizations for inference. Any such deviations should be recorded in the metadata or as
        // part of the architecture definition.
        //
        // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
        // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
        // should be padded to `ALIGNMENT` bytes.
        // uint8_t tensor_data[];
        this.tensorDataOffset = fileChannel.position();
        Arena arena = Arena.ofAuto();
        this.tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, fileChannel.size() - tensorDataOffset, arena);
        this.tensorEntries = HashMap.newHashMap(tensorInfos.size());
        for(Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet())
        {
            GGUFTensorInfo ti = entry.getValue();
            int numberOfElements = FloatTensor.numberOfElements(ti.dimensions());
            int sizeInBytes = Math.toIntExact(ti.ggmlType().byteSizeFor(numberOfElements));
            MemorySegment memorySegment = tensorData.asSlice(ti.offset(), sizeInBytes);
            tensorEntries.put(ti.name(), new GGUFTensorEntry(tensorData, ti.name(), ti.ggmlType(), ti.dimensions(), memorySegment));
        }
    }


    private GGUFType readGGMLType(FileChannel fileChannel) throws IOException
    {
        int ggmlTypeId = readInt(fileChannel); // ggml_type type;
        return GGUFType.fromId(ggmlTypeId);
    }


    private GGUFTensorInfo readTensorInfo(FileChannel fileChannel) throws IOException
    {
        // The name of the tensor. It is a standard GGUF string, with the caveat that
        // it must be at most 64 bytes long.
        String name = readString(fileChannel); // gguf_string_t name;
        assert name.length() <= 64;
        // The number of dimensions in the tensor.
        // Currently at most 4, but this may change in the future.
        int n_dimensions = readInt(fileChannel); // uint32_t n_dimensions;
        assert n_dimensions <= 4;
        // The dimensions of the tensor.
        int[] dimensions = new int[n_dimensions]; // uint64_t dimensions[n_dimensions];
        for(int i = 0; i < n_dimensions; ++i)
        {
            dimensions[i] = Math.toIntExact(readLong(fileChannel));
        }
        // The type of the tensor.
        GGUFType ggmlType = readGGMLType(fileChannel); // ggml_type type;
        // The offset of the tensor's data in this file in bytes.
        // This offset is relative to `tensor_data`, not to the start
        // of the file, to make it easier for writers to write the file.
        // Readers should consider exposing this offset relative to the
        // file to make it easier to read the data.
        // Must be a multiple of `ALIGNMENT`.
        long offset = readLong(fileChannel); // uint64_t offset;
        assert offset % getAlignment() == 0;
        return new GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }


    private String readString(FileChannel fileChannel) throws IOException
    {
        // A string in GGUF.
        // The length of the string, in bytes.
        int len = Math.toIntExact(readLong(fileChannel)); // uint64_t len;
        // The string as a UTF-8 non-null-terminated string.
        byte[] bytes = new byte[len]; // char string[len];
        int bytesRead = fileChannel.read(ByteBuffer.wrap(bytes));
        assert len == bytesRead;
        return new String(bytes, StandardCharsets.UTF_8);
    }


    private Pair<String, Object> readKeyValuePair(FileChannel fileChannel) throws IOException
    {
        // The key of the metadata. It is a standard GGUF string, with the following caveats:
        // - It must be a valid ASCII string.
        // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
        // - It must be at most 2^16-1/65535 bytes long.
        // Any keys that do not follow these rules are invalid.
        String key = readString(fileChannel); // gguf_string_t key;
        assert key.length() < (1 << 16);
        assert key.codePoints().allMatch(cp -> ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.');
        Object value = readMetadataValue(fileChannel);
        return new Pair<>(key, value);
    }


    private Object readMetadataValue(FileChannel fileChannel) throws IOException
    {
        // The type of the value.
        // Must be one of the `gguf_metadata_value_type` values.
        MetadataValueType value_type = readMetadataValueType(fileChannel); // gguf_metadata_value_type value_type;
        // The value.
        return readMetadataValueOfType(value_type, fileChannel); // gguf_metadata_value_t value;
    }


    void readHeader(FileChannel fileChannel) throws IOException
    {
        // Magic number to announce that this is a GGUF file.
        // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
        // Your executor might do little-endian byte order, so it might be
        // check for 0x46554747 and letting the endianness cancel out.
        // Consider being *very* explicit about the byte order here.
        this.magic = readInt(fileChannel); //    uint32_t magic;
        if(magic != GGUF_MAGIC)
        {
            throw new IllegalArgumentException("unsupported header.magic " + magic);
        }
        // The version of the format implemented.
        // Must be `3` for version described in this spec.
        //
        // This version should only be increased for structural changes to the format.
        // Changes that do not affect the structure of the file should instead update the metadata
        // to signify the change.
        this.version = readInt(fileChannel); // uint32_t version;
        if(!SUPPORTED_GGUF_VERSIONS.contains(version))
        {
            throw new IllegalArgumentException("unsupported header.version " + version);
        }
        // The number of tensors in the file.
        // This is explicit, instead of being included in the metadata, to ensure it is always present
        // for loading the tensors.
        this.tensorCount = Math.toIntExact(readLong(fileChannel)); // uint64_t tensor_count;
        // The number of metadata key-value pairs.
        this.metadata_kv_count = Math.toIntExact(readLong(fileChannel)); // uint64_t metadata_kv_count;
        // The metadata key-value pairs.
        // gguf_metadata_kv_t metadata_kv[metadata_kv_count];
        this.metadata = HashMap.newHashMap(metadata_kv_count);
        for(int i = 0; i < metadata_kv_count; ++i)
        {
            Pair<String, Object> keyValue = readKeyValuePair(fileChannel);
            assert !metadata.containsKey(keyValue.first());
            metadata.put(keyValue.first(), keyValue.second());
        }
    }


    private Object readArray(FileChannel fileChannel) throws IOException
    {
        // Any value type is valid, including arrays.
        MetadataValueType value_type = readMetadataValueType(fileChannel); // gguf_metadata_value_type type;
        // Number of elements, not bytes
        int len = Math.toIntExact(readLong(fileChannel)); // uint64_t len;
        // The array of values.
        // gguf_metadata_value_t array[len];
        switch(value_type)
        {
            case UINT8, INT8 ->
            {
                byte[] bytes = new byte[len];
                for(int i = 0; i < len; ++i)
                {
                    bytes[i] = readByte(fileChannel);
                }
                return bytes;
            }
            case UINT16, INT16 ->
            {
                short[] shorts = new short[len];
                for(int i = 0; i < len; ++i)
                {
                    shorts[i] = readShort(fileChannel);
                }
                return shorts;
            }
            case UINT32, INT32 ->
            {
                int[] ints = new int[len];
                for(int i = 0; i < len; ++i)
                {
                    ints[i] = readInt(fileChannel);
                }
                return ints;
            }
            case FLOAT32 ->
            {
                float[] floats = new float[len];
                for(int i = 0; i < len; ++i)
                {
                    floats[i] = readFloat(fileChannel);
                }
                return floats;
            }
            case BOOL ->
            {
                boolean[] booleans = new boolean[len];
                for(int i = 0; i < len; ++i)
                {
                    booleans[i] = readBoolean(fileChannel);
                }
                return booleans;
            }
            case STRING ->
            {
                String[] strings = new String[len];
                for(int i = 0; i < len; ++i)
                {
                    strings[i] = readString(fileChannel);
                }
                return strings;
            }
            case ARRAY ->
            {
                Object[] arrays = new Object[len];
                for(int i = 0; i < len; ++i)
                {
                    arrays[i] = readArray(fileChannel);
                }
                return arrays;
            }
            default -> throw new UnsupportedOperationException("read array of " + value_type);
        }
    }


    private Object readMetadataValueOfType(MetadataValueType valueType, FileChannel fileChannel) throws IOException
    {
        return switch(valueType)
        {
            case UINT8, INT8 -> readByte(fileChannel);
            case UINT16, INT16 -> readShort(fileChannel);
            case UINT32, INT32 -> readInt(fileChannel);
            case FLOAT32 -> readFloat(fileChannel);
            case UINT64, INT64 -> readLong(fileChannel);
            case FLOAT64 -> readDouble(fileChannel);
            case BOOL -> readBoolean(fileChannel);
            case STRING -> readString(fileChannel);
            case ARRAY -> readArray(fileChannel);
        };
    }


    private byte readByte(FileChannel fileChannel) throws IOException
    {
        int bytesRead = fileChannel.read(BB_1);
        assert bytesRead == 1;
        return BB_1.clear().get(0);
    }


    private boolean readBoolean(FileChannel fileChannel) throws IOException
    {
        return readByte(fileChannel) != 0;
    }


    private short readShort(FileChannel fileChannel) throws IOException
    {
        int bytesRead = fileChannel.read(BB_2);
        assert bytesRead == 2;
        return BB_2.clear().getShort(0);
    }


    private int readInt(FileChannel fileChannel) throws IOException
    {
        int bytesRead = fileChannel.read(BB_4);
        assert bytesRead == 4;
        return BB_4.clear().getInt(0);
    }


    private long readLong(FileChannel fileChannel) throws IOException
    {
        int bytesRead = fileChannel.read(BB_8);
        assert bytesRead == 8;
        return BB_8.clear().getLong(0);
    }


    private float readFloat(FileChannel fileChannel) throws IOException
    {
        return Float.intBitsToFloat(readInt(fileChannel));
    }


    private double readDouble(FileChannel fileChannel) throws IOException
    {
        return Double.longBitsToDouble(readLong(fileChannel));
    }


    private MetadataValueType readMetadataValueType(FileChannel fileChannel) throws IOException
    {
        int index = readInt(fileChannel);
        return MetadataValueType.fromIndex(index);
    }


    public int getAlignment()
    {
        if(alignment != 0)
        {
            return alignment;
        }
        alignment = (int)metadata.getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
        assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        return alignment;
    }
}
