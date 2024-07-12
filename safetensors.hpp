#ifndef SAFETENSORS_HPP
#define SAFETENSORS_HPP

// Constants
#define SAFETENSORS_MAX_DIM 8
#define SAFETENSORS_MAX_TENSORS 2048
#define SAFETENSORS_MAX_FILE_SIZE (2ULL << 40) // 2 TiB
#define SAFETENSORS_MAX_STRING_SIZE 2048
#define SAFETENSORS_MAX_METADATA_SIZE 8192

#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>
#include <fstream>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>
#include <climits>

namespace safetensors
{

    // Exception class
    class SafetensorsException : public std::runtime_error
    {
    public:
        explicit SafetensorsException(const std::string &message) : std::runtime_error(message) {}
    };

    // Struct definitions
    struct TensorInfo
    {
        std::string dtype;
        std::vector<int64_t> shape;
        std::array<size_t, 2> data_offsets;
    };

    // Function declarations
    inline torch::ScalarType get_torch_dtype(const std::string &dtype_str);
    inline std::string get_safetensors_dtype(torch::ScalarType dtype);
    inline void validate_string_length(const std::string &str, const std::string &context);
    inline bool is_big_endian();

    template <typename T>
    inline T swap_endian(T u);

    // Class definitions
    class SimpleJSONParser
    {
    private:
        const char *json;
        size_t pos;

        inline void skipWhitespace()
        {
            while (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')
                pos++;
        }

        inline std::string parseString()
        {
            std::string result;
            pos++; // Skip opening quote
            while (json[pos] != '"')
            {
                if (json[pos] == '\\')
                {
                    pos++;
                    if (json[pos] == 'u')
                    {
                        // Handle Unicode escape (simplified)
                        pos += 4;
                    }
                }
                result += json[pos++];
            }
            pos++; // Skip closing quote
            return result;
        }

        inline std::vector<int64_t> parseArray()
        {
            std::vector<int64_t> result;
            pos++; // Skip opening bracket
            while (json[pos] != ']')
            {
                skipWhitespace();
                size_t num_start = pos;
                while (std::isdigit(json[pos]))
                    pos++;
                result.push_back(std::stoll(std::string(json + num_start, pos - num_start)));
                skipWhitespace();
                if (json[pos] == ',')
                    pos++;
            }
            pos++; // Skip closing bracket
            return result;
        }

        inline std::array<size_t, 2> parseDataOffsets()
        {
            std::array<size_t, 2> result;
            pos++; // Skip opening bracket
            skipWhitespace();
            size_t num_start = pos;
            while (std::isdigit(json[pos]))
                pos++;
            result[0] = std::stoull(std::string(json + num_start, pos - num_start));
            skipWhitespace();
            pos++; // Skip comma
            skipWhitespace();
            num_start = pos;
            while (std::isdigit(json[pos]))
                pos++;
            result[1] = std::stoull(std::string(json + num_start, pos - num_start));
            skipWhitespace();
            pos++; // Skip closing bracket
            return result;
        }

        inline TensorInfo parseTensorInfo()
        {
            TensorInfo info;
            pos++; // Skip opening brace
            while (json[pos] != '}')
            {
                skipWhitespace();
                std::string key = parseString();
                skipWhitespace();
                pos++; // Skip colon
                skipWhitespace();
                if (key == "dtype")
                {
                    info.dtype = parseString();
                }
                else if (key == "shape")
                {
                    info.shape = parseArray();
                }
                else if (key == "data_offsets")
                {
                    info.data_offsets = parseDataOffsets();
                }
                else
                {
                    // Skip unknown fields
                    while (json[pos] != ',' && json[pos] != '}')
                        pos++;
                }
                skipWhitespace();
                if (json[pos] == ',')
                    pos++;
            }
            pos++; // Skip closing brace
            return info;
        }

    public:
        inline SimpleJSONParser(const char *json_str) : json(json_str), pos(0) {}

        inline std::unordered_map<std::string, TensorInfo> parse()
        {
            std::unordered_map<std::string, TensorInfo> result;
            skipWhitespace();
            if (json[pos++] != '{')
                throw SafetensorsException("Expected object");
            while (json[pos] != '}')
            {
                skipWhitespace();
                std::string key = parseString();
                skipWhitespace();
                pos++; // Skip colon
                skipWhitespace();
                if (key != "__metadata__")
                {
                    result[key] = parseTensorInfo();
                }
                else
                {
                    // Skip metadata
                    while (json[pos] != ',' && json[pos] != '}')
                        pos++;
                }
                skipWhitespace();
                if (json[pos] == ',')
                    pos++;
            }
            return result;
        }
    };

    // Function implementations
    inline torch::ScalarType get_torch_dtype(const std::string &dtype_str)
    {
        static const std::unordered_map<std::string, torch::ScalarType> dtype_map = {
            {"BOOL", torch::kBool},
            {"U8", torch::kUInt8},
            {"I8", torch::kInt8},
            {"U16", torch::kUInt16},
            {"I16", torch::kInt16},
            {"U32", torch::kUInt32},
            {"I32", torch::kInt32},
            {"U64", torch::kUInt64},
            {"I64", torch::kInt64},
            {"F16", torch::kFloat16},
            {"BF16", torch::kBFloat16},
            {"F32", torch::kFloat32},
            {"F64", torch::kFloat64}};

        auto it = dtype_map.find(dtype_str);
        if (it != dtype_map.end())
        {
            return it->second;
        }
        throw SafetensorsException("Unknown dtype: " + dtype_str);
    }

    inline std::string get_safetensors_dtype(torch::ScalarType dtype)
    {
        static const std::unordered_map<torch::ScalarType, std::string> dtype_map = {
            {torch::kBool, "BOOL"},
            {torch::kUInt8, "U8"},
            {torch::kInt8, "I8"},
            {torch::kUInt16, "U16"},
            {torch::kInt16, "I16"},
            {torch::kUInt32, "U32"},
            {torch::kInt32, "I32"},
            {torch::kUInt64, "U64"},
            {torch::kInt64, "I64"},
            {torch::kFloat16, "F16"},
            {torch::kBFloat16, "BF16"},
            {torch::kFloat32, "F32"},
            {torch::kFloat64, "F64"}};

        auto it = dtype_map.find(dtype);
        if (it != dtype_map.end())
        {
            return it->second;
        }
        throw SafetensorsException("Unsupported dtype");
    }

    inline std::unordered_map<std::string, TensorInfo> parse_safetensors_header_info(const char *data, size_t size)
    {
        if (size < 8)
            throw SafetensorsException("Invalid file size");

        uint64_t header_size;
        std::memcpy(&header_size, data, sizeof(uint64_t));

        if (8 + header_size > size)
            throw SafetensorsException("Invalid header size");

        SimpleJSONParser parser(data + 8);
        return parser.parse();
    }

    inline void validate_string_length(const std::string &str, const std::string &context)
    {
        if (str.length() > SAFETENSORS_MAX_STRING_SIZE)
        {
            throw SafetensorsException(context + " exceeds maximum allowed length");
        }
    }

    inline bool is_big_endian()
    {
        union
        {
            uint32_t i;
            char c[4];
        } bint = {0x01020304};

        return bint.c[0] == 1;
    }

    template <typename T>
    inline T swap_endian(T u)
    {
        static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");

        union
        {
            T u;
            unsigned char u8[sizeof(T)];
        } source, dest;

        source.u = u;

        for (size_t k = 0; k < sizeof(T); k++)
            dest.u8[k] = source.u8[sizeof(T) - k - 1];

        return dest.u;
    }

    inline std::unordered_map<std::string, torch::Tensor> load_safetensors(const std::string &filename, torch::Device device)
    {
        int fd = open(filename.c_str(), O_RDONLY);
        if (fd == -1)
        {
            throw SafetensorsException("Failed to open file: " + filename);
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1)
        {
            close(fd);
            throw SafetensorsException("Failed to get file size");
        }
        size_t file_size = sb.st_size;

        if (file_size > SAFETENSORS_MAX_FILE_SIZE)
        {
            close(fd);
            throw SafetensorsException("File size exceeds maximum allowed size");
        }

        void *mapped_file = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped_file == MAP_FAILED)
        {
            close(fd);
            throw SafetensorsException("Failed to memory map file");
        }

        try
        {
            uint64_t header_size;
            std::memcpy(&header_size, mapped_file, sizeof(uint64_t));
            if (is_big_endian())
            {
                header_size = swap_endian(header_size);
            }

            if (8 + header_size > file_size)
                throw SafetensorsException("Invalid header size");

            auto tensor_infos = parse_safetensors_header_info(static_cast<char *>(mapped_file), file_size);

            if (tensor_infos.size() > SAFETENSORS_MAX_TENSORS)
            {
                throw SafetensorsException("Number of tensors exceeds maximum allowed");
            }

            std::unordered_map<std::string, torch::Tensor> tensors;
            char *data_start = static_cast<char *>(mapped_file) + 8 + header_size;

            for (const auto &[name, info] : tensor_infos)
            {
                validate_string_length(name, "Tensor name");

                if (info.shape.size() > SAFETENSORS_MAX_DIM)
                {
                    throw SafetensorsException("Tensor dimension exceeds maximum allowed");
                }

                torch::ScalarType dtype = get_torch_dtype(info.dtype);

                auto options = torch::TensorOptions()
                                   .dtype(dtype)
                                   .device(torch::kCPU);

                torch::Tensor cpu_tensor = torch::from_blob(
                                               data_start + info.data_offsets[0],
                                               info.shape,
                                               options)
                                               .clone(); // Clone to own the data

                if (is_big_endian() && (dtype == torch::kFloat16 || dtype == torch::kFloat32 || dtype == torch::kFloat64))
                {
                    auto data_ptr = static_cast<char *>(cpu_tensor.data_ptr());
                    for (int64_t i = 0; i < cpu_tensor.numel() * cpu_tensor.element_size(); i += cpu_tensor.element_size())
                    {
                        std::reverse(data_ptr + i, data_ptr + i + cpu_tensor.element_size());
                    }
                }

                tensors[name] = cpu_tensor.to(device);
            }

            munmap(mapped_file, file_size);
            close(fd);

            return tensors;
        }
        catch (...)
        {
            munmap(mapped_file, file_size);
            close(fd);
            throw;
        }
    }

    inline void save_safetensors(const std::unordered_map<std::string, torch::Tensor> &tensors, const std::string &filename, const std::unordered_map<std::string, std::string> &metadata = {})
    {
        if (tensors.size() > SAFETENSORS_MAX_TENSORS)
        {
            throw SafetensorsException("Number of tensors exceeds maximum allowed");
        }

        std::string header_json = "{";
        std::vector<char> data_buffer;
        size_t current_offset = 0;

        if (!metadata.empty())
        {
            header_json += "\"__metadata__\":{";
            bool first_meta = true;
            for (const auto &[key, value] : metadata)
            {
                validate_string_length(key, "Metadata key");
                validate_string_length(value, "Metadata value");

                if (!first_meta)
                    header_json += ",";
                header_json += "\"" + key + "\":\"" + value + "\"";
                first_meta = false;
            }
            header_json += "},";
        }

        for (const auto &[name, tensor] : tensors)
        {
            validate_string_length(name, "Tensor name");

            auto cpu_tensor = tensor.to(torch::kCPU).contiguous();

            if (cpu_tensor.dtype() == torch::kFloat16 || cpu_tensor.dtype() == torch::kFloat32 || cpu_tensor.dtype() == torch::kFloat64)
            {
                cpu_tensor = cpu_tensor.to(torch::kCPU, cpu_tensor.dtype(), /*non_blocking=*/false, /*copy=*/true);
                auto data_ptr = static_cast<char *>(cpu_tensor.data_ptr());
                for (int64_t i = 0; i < cpu_tensor.numel() * cpu_tensor.element_size(); i += cpu_tensor.element_size())
                {
                    std::reverse(data_ptr + i, data_ptr + i + cpu_tensor.element_size());
                }
            }

            if (cpu_tensor.dim() > SAFETENSORS_MAX_DIM)
            {
                throw SafetensorsException("Tensor dimension exceeds maximum allowed");
            }

            auto dtype = get_safetensors_dtype(cpu_tensor.scalar_type());
            auto shape = cpu_tensor.sizes().vec();
            size_t tensor_size = cpu_tensor.numel() * cpu_tensor.element_size();

            if (header_json.length() > 1)
                header_json += ",";
            header_json += "\"" + name + "\":{";
            header_json += "\"dtype\":\"" + dtype + "\",";
            header_json += "\"shape\":[";
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (i > 0)
                    header_json += ",";
                header_json += std::to_string(shape[i]);
            }
            header_json += "],";
            header_json += "\"data_offsets\":[" + std::to_string(current_offset) + "," + std::to_string(current_offset + tensor_size) + "]";
            header_json += "}";

            const char *tensor_data = static_cast<const char *>(cpu_tensor.data_ptr());
            data_buffer.insert(data_buffer.end(), tensor_data, tensor_data + tensor_size);

            current_offset += tensor_size;
        }

        header_json += "}";
        uint64_t header_size = header_json.size();

        if (header_size > SAFETENSORS_MAX_METADATA_SIZE)
        {
            throw SafetensorsException("Metadata size exceeds maximum allowed size");
        }

        if (8 + header_size + data_buffer.size() > SAFETENSORS_MAX_FILE_SIZE)
        {
            throw SafetensorsException("Total file size exceeds maximum allowed size");
        }

        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            throw SafetensorsException("Failed to open file for writing: " + filename);
        }

        uint64_t little_endian_header_size = header_size;
        if (is_big_endian())
        {
            little_endian_header_size = swap_endian(header_size);
        }
        file.write(reinterpret_cast<const char *>(&little_endian_header_size), sizeof(uint64_t));

        file.write(header_json.data(), header_json.size());

        file.write(data_buffer.data(), data_buffer.size());

        if (!file)
        {
            throw SafetensorsException("Failed to write to file: " + filename);
        }
    }

} // namespace safetensors

#endif // SAFETENSORS_HPP