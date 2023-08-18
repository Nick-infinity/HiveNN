# Hive Neural Network (HNN)
Mini Neural Network Framework to execute large language model on edge devices. Currently support llama 2 / GPTmodels
## Requirements
* Python3
* Android NDK and Android SDK Pltform tools
* Bazel 5.3.0 (check bazelisk installation)

## Build

### HNN Test App
* Hnn Test app links with libhhn.so statically 
- Release build for android
```
bazel build -c opt  --config android_arm64 :Hnn_Test
```
- Debug build for android
```
bazel build -c opt  --copt="-DDEBUG_BUILD" --config android_arm64 :Hnn_Test
```

### Quantizer
```
bazel build :quantizer
```

## Usage
* Build Hnn_Test app
* Build Build Quantizer
* Download Llama_2_chat model from hugging face
* Execute the model conversion script on 
```
python create_meta_llama_bin.py $path_to_model_directory
```
* Quantize the model using quantizer
```
./bazel-build/quantizer -f $path_to_converted_model -g $export_model_path -n 0
```
* Place the tokenizer.bin quantized_model.bin & Hnn_Test in /data/local/tmp/ of adb shell
* Give permission
```
cd /data/local/tmp
chmod 777 *
```
* Execute the model
```
./Hnn_Test quantized_model.bin tokenizer.bin "Your Propmt"
```

