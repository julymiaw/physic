# 仓库说明

```bash
g++ -I/usr/include/opencv4 -o 轮廓检测 轮廓检测.cpp -lopencv_core -lopencv_imgproc -lopencv_videoio -lcurl
```

```bash
emcc -I/usr/include/opencv4 -I/home/july/physic/test/opencv-4.x/build_wasm/include -L/home/july/physic/test/opencv-4.x/build_wasm/lib -o 轮廓检测.js 轮廓检测.cpp -lopencv_core -lopencv_imgproc -lembind -s WASM=1 -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap"]' -sALLOW_MEMORY_GROWTH -sNO_DISABLE_EXCEPTION_CATCHING
```
