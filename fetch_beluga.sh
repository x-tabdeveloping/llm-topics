mkdir generative_models
cd generative_models/
echo "Fethcing Beluga-7B quantized..."
wget "https://huggingface.co/TheBloke/StableBeluga-7B-GGUF/resolve/main/stablebeluga-7b.Q5_K_M.gguf"
echo "DONE"
cd ..

