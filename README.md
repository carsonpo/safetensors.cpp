# Safetensors.cpp

> Zero<sup>*</sup> Dependency Safetensors Loading and Storing with C++ and LibTorch

## Arbitrary Limits

- 2TiB overall file size (enough to store LLaMA 405b at full fp32 precision)
- 2048 tensors
- 8 dims per tensor
- 2KiB max string size in metadata
- 8KiB max overall metadata size

If you want to change these, they're in the header.

## Usage Examples

### Loading Safetensors

```cpp
#include "safetensors.hpp"

int main() {
    try {
        // Load tensors from a file
        std::string filename = "model.safetensors";
        auto tensors = safetensors::load_safetensors(filename);

        // Print information about loaded tensors
        for (const auto& [name, tensor] : tensors) {
            std::cout << "Tensor name: " << name << std::endl;
            std::cout << "Shape: " << tensor.sizes() << std::endl;
            std::cout << "Dtype: " << tensor.dtype() << std::endl;
            std::cout << std::endl;
        }
    } catch (const safetensors::SafetensorsException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
```

### Saving Safetensors

```cpp
#include "safetensors.hpp"

int main() {
    try {
        // Create some example tensors
        std::unordered_map<std::string, torch::Tensor> tensors;
        tensors["weight"] = torch::randn({3, 3});
        tensors["bias"] = torch::zeros({3});

        // Add some metadata
        std::unordered_map<std::string, std::string> metadata;
        metadata["description"] = "Example model";
        metadata["version"] = "1.0";

        // Save tensors to a file
        std::string filename = "model_output.safetensors";
        safetensors::save_safetensors(tensors, filename, metadata);

        std::cout << "Tensors saved successfully to " << filename << std::endl;
    } catch (const safetensors::SafetensorsException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
```

### Loading to a Specific Device

```cpp
#include "safetensors.hpp"

int main() {
    try {
        std::string filename = "model.safetensors";
        torch::Device device(torch::kCUDA);  // or torch::kCPU for CPU
        auto tensors = safetensors::load_safetensors(filename, device);

        // Tensors are now loaded directly to the specified device
        for (const auto& [name, tensor] : tensors) {
            std::cout << "Tensor " << name << " is on " << tensor.device() << std::endl;
        }
    } catch (const safetensors::SafetensorsException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
```

## License

MIT

---

<sup>*</sup> LibTorch is required, but no additional dependencies are needed for safetensors functionality.
