:: Build CPU-only version
cmake -DCUDA_DISABLED=True .
cmake --build . --config Release
:: Build CPU+CUDA version
cmake -DCUDA_DISABLED=OFF .
cmake --build . --config Release
pause