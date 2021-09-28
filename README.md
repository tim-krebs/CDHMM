# Speech Recognition with Continious Density Hidden Markov Modells

![C++](https://camo.githubusercontent.com/c59efb57803dde7f352f4932a468a7f39fa2fb5f/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f632532422532422d31312f31342f31372f32302d626c75652e737667)
![License](https://camo.githubusercontent.com/890acbdcb87868b382af9a4b1fac507b9659d9bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c75652e737667)

## Structure

```text
├── CMakeLists.txt
├── app
│   ├── CMakesLists.txt
│   └── main.cc
├── benchmarks
│   ├── CMakesLists.txt
│   └── main.cc
├── docs
├── ├── Doxyfile
│   └── html/
├── external
│   ├── CMakesLists.txt
│   ├── linalg/
│   └── loguru/
├── include
│   └── my_lib.h
├── src
│   ├── CMakesLists.txt
│   └── my_lib.cc
└── tests
  ├── CMakeLists.txt
  └── main.cc
```

## Software Requirements

- CMake 3.14+
- Code Covergae (only on GNU|Clang): lcov, gcovr
